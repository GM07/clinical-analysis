import os
import random
from datasets import load_dataset, load_from_disk
import pandas as pd
from tqdm import tqdm

from src.data.synthetic_dataset import SyntheticDataset

class MedMCQA(SyntheticDataset):

    DEFAULT_SYSTEM_PROMPT = 'Given a question and an answer, your role is to transform the question into a statement by incorporating the answer with it. Do not add any details that is not mentioned in the question or the answer.'
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

        one_shot_conversation = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': one_shot_question},
            {'role': 'assistant', 'content': one_shot_answer}
        ]

        data = self.prepare_for_prompting()
        
        prompts = []

        for row in tqdm(data, desc='Generating prompts', total=len(data)):
            id = row['id']
            question = row['question']
            correct = row['correct_answer']
            wrong = row['random_wrong_answer']
            explanation = row['explanation']
            correct_question = f"""Question: {question}\nAnswer: {correct}"""
            wrong_question = f"""Question: {question}\nAnswer: {wrong}"""
            correct_prompt = {
                'role': 'user',
                'content': correct_question
            }

            wrong_prompt = {
                'role': 'user',
                'content': wrong_question
            }

            one_shot_correct = one_shot_conversation + [correct_prompt]
            one_shot_wrong = one_shot_conversation + [wrong_prompt]

            prompts.append((id, one_shot_correct, question, correct, True, explanation, system_prompt, one_shot_question, one_shot_answer, correct_question))
            prompts.append((id, one_shot_wrong, question, wrong, False, explanation, system_prompt, one_shot_question, one_shot_answer, wrong_question))

        df = pd.DataFrame(prompts, columns=['id', 'chat', 'question', 'answer', 'is_correct', 'explanation', 'system_prompt', 'one_shot_user_input', 'one_shot_assistant_output', 'user_input'])

        if output_path:
            df.to_csv(output_path, index=False)

        return df
