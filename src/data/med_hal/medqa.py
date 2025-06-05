from collections import defaultdict
import random
import re
from datasets import Dataset
import pandas as pd
from tqdm import tqdm

from src.data.synthetic_dataset import SyntheticDataset

class MedQA(SyntheticDataset):

    QUESTION_PATTERN = r'^(.*?)([^.!?]+\?)$'
    DEFAULT_SYSTEM_PROMPT = 'Given a medical text, a question about the text and the associated answer, your role is to transform the question into a statement by incorporating the answer with it. Do not add any details that is not mentioned in the question or the answer.'
    DEFAULT_ONE_SHOT_USER = 'Question: Which of the following is the best treatment for this patient?\nAnswer: Nitrofurantoin'
    DEFAULT_ONE_SHOT_ASSISTANT = 'Nitrofurantoin is the best treatment for this patient.'

    def __init__(self, path: str):
        self.path = path if path.endswith('/') else path + '/'
        self.data_path = self.path + 'questions/US/train.jsonl'

        self.load()

    def load(self):
        self.data = Dataset.from_json(self.data_path)

    def prepare_for_prompting(self):
        """
        Prepares the dataset in the form of a dataframe that contains four columns: 
        - question: Question about the medical domain
        - correct_answer: Correct answer to the question
        - random_wrong_answer: Random wrong answer to the question
        - explanation: Explanation of the correct answer

        This dataframe can be used to generate prompts for the model that will generate
        a factual or hallucinated statement about the medical domain.

        """

        def get_answers(row):
            # Split question into context and question
            text = row['question']
            match = re.search(self.QUESTION_PATTERN, text, re.DOTALL)
            if match:
                context = match.group(1)
                question = match.group(2)
            else:
                return {
                    'context': None,
                    'question': None,
                    'correct_answer': None,
                    'random_wrong_answer': None,
                    'explanation': None
                }

            # Map the options to their actual answers
            options = row['options']
            
            # Get the correct answer using the 'cop' value
            correct_answer = options[str(row['answer_idx'])]
            
            # Get all wrong answers
            wrong_answers = [ans for opt, ans in options.items() if opt != str(row['answer_idx'])]
            
            # Select a random wrong answer
            random_wrong = random.choice(wrong_answers)
            
            return {
                'context': context,
                'question': question,
                'correct_answer': correct_answer,
                'random_wrong_answer': random_wrong,
                'explanation': None
            }

        def generate_positive_and_negative_samples(row):
            # Batch size must be 1
            out = {
                'context': [row['context'][0], row['context'][0]],
                'question': [row['question'][0], row['question'][0]],
                'answer': [row['correct_answer'][0], row['random_wrong_answer'][0]],
                'is_correct': [True, False],
                'explanation': [None, None],
                'options': [row['options'][0], row['options'][0]],
                'correct_answer': [row['correct_answer'][0], row['correct_answer'][0]]
            }

            return out

        self.data = self.data.map(get_answers)
        self.data = self.data.filter(lambda x: x['question'] is not None)
        self.data = self.data.map(
            generate_positive_and_negative_samples, 
            remove_columns=self.data.column_names,
            batch_size=1, 
            batched=True
        )

        return self.data
    
    def generate_prompts(
        self, 
        system_prompt: str = None,
        one_shot_question: str = None,
        one_shot_answer: str = None,
        output_path: str = None
    ):
        """
        Generates prompts for the model that will be used to generate a factual or hallucinated statement
        about the medical domain. This function will generate two prompts for each question in the dataset:
        - One prompt with the correct answer
        - One prompt with a random wrong answer

        Args:
            system_prompt: The system prompt to use for the model. Will override the default system prompt.
            one_shot_question: The one-shot question to use for the model. Will override the default one-shot question.
            one_shot_answer: The one-shot answer to use for the model. Will override the default one-shot answer.
            output_path: The path to save the prompts to. If None, the prompts will not be saved.
        """

        system_prompt = system_prompt if system_prompt else self.DEFAULT_SYSTEM_PROMPT
        one_shot_question = one_shot_question if one_shot_question else self.DEFAULT_ONE_SHOT_USER
        one_shot_answer = one_shot_answer if one_shot_answer else self.DEFAULT_ONE_SHOT_ASSISTANT

        # one_shot_conversation = [
        #     {'role': 'system', 'content': system_prompt},
        #     {'role': 'user', 'content': one_shot_question},
        #     {'role': 'assistant', 'content': one_shot_answer}
        # ]

        data = self.prepare_for_prompting()
        
        def generate_positive_negative_samples(row):

            final_dict = defaultdict(list)
            for i in range(len(row['question'])):
                context = row['context'][i]
                question = row['question'][i]
                answer = row['answer'][i]
                is_correct = row['is_correct'][i]
                explanation = row['explanation'][i]
                options = row['options'][i]
                correct_answer = row['correct_answer'][i]
                user_input = f'Question: {question}\nAnswer: {answer}'

                # chat = one_shot_conversation + [
                #     {'role': 'user', 'content': user_input}
                # ]

                # rows.append((id, chat, context, question, answer, is_correct, explanation, system_prompt, one_shot_question, one_shot_answer, user_input))
                # columns=['id', 'chat', 'context', 'question', 'answer', 'is_correct', 'explanation', 'system_prompt', 'one_shot_user_input', 'one_shot_assistant_output', 'user_input']
                final_dict['id'].append(i)
                final_dict['context'].append(context)
                final_dict['options'].append(options)
                final_dict['correct_answer'].append(correct_answer)
                final_dict['question'].append(question)
                final_dict['answer'].append(answer)
                final_dict['is_correct'].append(is_correct)
                final_dict['explanation'].append(explanation)
                final_dict['system_prompt'].append(system_prompt)
                final_dict['one_shot_user_input'].append(one_shot_question)
                final_dict['one_shot_assistant_output'].append(one_shot_answer)
                final_dict['user_input'].append(user_input)

                # id = i
                # question = row['question'][i]
                # correct = row['correct_answer'][i]
                # wrong = row['random_wrong_answer'][i]
                # explanation = row['explanation'][i]
                # correct_question = f"""Question: {question}\nAnswer: {correct}"""
                # wrong_question = f"""Question: {question}\nAnswer: {wrong}"""
                # # correct_prompt = {
                # #     'role': 'user',
                # #     'content': correct_question
                # # }

                # # wrong_prompt = {
                # #     'role': 'user',
                # #     'content': wrong_question
                # # }

                # # one_shot_correct = one_shot_conversation + [correct_prompt]
                # # one_shot_wrong = one_shot_conversation + [wrong_prompt]
                
                # # return {
                # final_dict['id'].extend([id, id])
                
                # # final_dict['chat'].extend([one_shot_correct, one_shot_wrong])
                # final_dict['question'].extend([question, question])
                # final_dict['answer'].extend([correct, wrong])
                # final_dict['is_correct'].extend([True, False])
                # final_dict['explanation'].extend([explanation, explanation])
                # final_dict['system_prompt'].extend([system_prompt, system_prompt])
                # final_dict['one_shot_user_input'].extend([one_shot_question, one_shot_question])
                # final_dict['one_shot_assistant_output'].extend([one_shot_answer, one_shot_answer])
                # final_dict['user_input'].extend([correct_question, wrong_question])
                # }

            return final_dict
        df = data.map(generate_positive_negative_samples, batched=True, remove_columns=data.column_names)



        # rows = []

        # for id, row in tqdm(enumerate(data), desc='Generating prompts', total=len(data)):
        #     context = row['context']
        #     question = row['question']
        #     answer = row['answer']
        #     is_correct = row['is_correct']
        #     explanation = row['explanation']
        #     user_input = f'Question: {question}\nAnswer: {answer}'

        #     chat = one_shot_conversation + [
        #         {'role': 'user', 'content': user_input}
        #     ]

        #     rows.append((id, chat, context, question, answer, is_correct, explanation, system_prompt, one_shot_question, one_shot_answer, user_input))

        # df = pd.DataFrame(
        #     rows, 
        #     columns=['id', 'chat', 'context', 'question', 'answer', 'is_correct', 'explanation', 'system_prompt', 'one_shot_user_input', 'one_shot_assistant_output', 'user_input']
        # )

        if output_path:
            df.to_csv(output_path, index=False)

        return df


class MedQAValidator:

    PROMPT_TEMPLATE = """Your task is to evaluate whether a statement is factual or not. The statement is related to a question that was asked to a model about a certain context. You will be given the question and the correct answer. Based on the provided question and associated answer, you must evaluate whether the statement is factual or not. If the statement is factual, answer YES. If it is not, answer NO. If the model did not generate a statement, changes ideas in between sentences, does not generate a statement related to the question or states multiple things in the statement of which some are not factual, answer with NO.
Here is the context : {context}
Here is the question : {question}
Here is the correct answer to the question : {answer}
Here is the statement : {statement}

Only answer with YES or NO. Do not generate any other explanation."""


    def __init__(self, processed_path: str):
        self.processed_path = processed_path

        self.load()

    def load(self):
        self.dataset = Dataset.from_csv(self.processed_path)

    def generate_validation_prompts(self):

        def generate(x, template: str):
            return {'prompt': [template.format(context=context, question=question, answer=answer, statement=statement) for context, question, answer, statement in zip(x['context'], x['question'], x['correct_answer'], x['output'])]}

        columns = ['system_prompt', 'one_shot_user_input', 'one_shot_assistant_output', 'user_input']
        return self.dataset.map(generate, remove_columns=columns, batched=True, fn_kwargs={'template': self.PROMPT_TEMPLATE})
