from collections import defaultdict
import os
import random
from datasets import load_dataset, load_from_disk, Dataset

from src.data.synthetic_dataset import SyntheticDataset

class MedMCQA(SyntheticDataset):

    DEFAULT_SYSTEM_PROMPT = 'Given a question and an answer, your role is to transform multiple-choice questions into statements by incorporating their respective answers in it. Do not add any details that is not mentioned in the question or the answer.'
    DEFAULT_ONE_SHOT_USER = 'Question: Which of the following agents is most commonly associated with recurrent meningitis due to CSF leaks?\nAnswer: Pneumococci'
    DEFAULT_ONE_SHOT_ASSISTANT = 'Pneumococci is most commonly associated with recurrent meningitis due to CSF leaks'

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.load()

    def load(self):
        self.dataset_name = os.path.basename(self.dataset_path)
        self.dataset_folder_path = self.dataset_path.replace(self.dataset_name, '')

        # Load all splits and concatenate into a single dataset
        self.data = load_from_disk(
            self.dataset_path,
            # split='train+validation',
            # trust_remote_code=True
        )['train']

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
            # Map the options to their actual answers
            options = {
                '0': row['opa'],
                '1': row['opb'],
                '2': row['opc'],
                '3': row['opd']
            }
            
            # Get the correct answer using the 'cop' value
            correct_answer = options[str(row['cop'])]
            
            # Get all wrong answers
            wrong_answers = [ans for opt, ans in options.items() if opt != str(row['cop'])]
            
            # Select a random wrong answer
            random_wrong = random.choice(wrong_answers)
            
            return {
                'question': row['question'],
                'correct_answer': correct_answer,
                'random_wrong_answer': random_wrong,
                'explanation': row['exp']
            }
        
        def filter_invalid_samples(row):
            if 'the above' in row['opd'].lower():
                return False
            
            if 'except' in row['question']:
                return False

            for option in [row['opa'], row['opb'], row['opc'], row['opd']]:
                if 'true' in option.lower() or 'false' in option.lower():
                    return False
        
            return True
        
        self.data = self.data.filter(filter_invalid_samples)
        return self.data.map(get_answers)

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

        data = self.prepare_for_prompting()
        
        def generate_positive_negative_samples(row):

            final_dict = defaultdict(list)
            for i in range(len(row['id'])):
                id = row['id'][i]
                question = row['question'][i]
                correct = row['correct_answer'][i]
                wrong = row['random_wrong_answer'][i]
                explanation = row['explanation'][i]
                correct_question = f"""Question: {question}\nAnswer: {correct}"""
                wrong_question = f"""Question: {question}\nAnswer: {wrong}"""

                final_dict['id'].extend([id, id])
                final_dict['question'].extend([question, question])
                final_dict['answer'].extend([correct, wrong])
                final_dict['is_correct'].extend([True, False])
                final_dict['correct_answer'].extend([correct, correct])
                final_dict['explanation'].extend([explanation, explanation])
                final_dict['system_prompt'].extend([system_prompt, system_prompt])
                final_dict['one_shot_user_input'].extend([one_shot_question, one_shot_question])
                final_dict['one_shot_assistant_output'].extend([one_shot_answer, one_shot_answer])
                final_dict['user_input'].extend([correct_question, wrong_question])

            return final_dict
        df = data.map(generate_positive_negative_samples, batched=True, remove_columns=data.column_names)

        if output_path:
            df.to_csv(output_path, index=False)

        return df

class MedMCQAValidator:

    PROMPT_TEMPLATE = """Your task is to evaluate whether a statement is factual or not. The statement is related to a question that was asked to a model. You will be given the question and the correct answer. Based on the provided question and associated answer, you must evaluate whether the statement is factual or not. If the statement is factual, answer YES. If it is not, answer NO. If the model did not generate a statement, changes ideas in between sentences, does not generate a statement related to the question or states multiple things in the statement of which some are not factual, answer with NO.
Here is the question : {question}
Here is the correct answer to the question : {answer}
Here is the statement : {statement}

Only answer with YES or NO. Do not generate any other explanation."""


    def __init__(self, processed_path: str):
        self.processed_path = processed_path

        self.load()

    def load(self):

        if self.processed_path.endswith('.csv'):
            self.dataset = Dataset.from_csv(self.processed_path)
        else:
            self.dataset = load_from_disk(self.processed_path)

    def generate_validation_prompts(self):

        def generate(x, template: str):
            return {'prompt': [template.format(question=question, answer=answer, statement=statement) for question, answer, statement in zip(x['question'], x['correct_answer'], x['output'])]}

        columns = ['system_prompt', 'one_shot_user_input', 'one_shot_assistant_output', 'user_input']
        return self.dataset.map(generate, remove_columns=columns, batched=True, fn_kwargs={'template': self.PROMPT_TEMPLATE})
