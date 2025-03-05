
import uuid
from datasets import Dataset as HuggingFaceDataset, concatenate_datasets
import nltk
import nltk.translate.bleu_score

from src.data.augmented_clinical_notes import AugmentedClinicalNotes

class MedHal:

    """
    The MedHal dataset is a dataset of medical statements and their explanations.
    It is used to train and evaluate medical statement verification models.

    It contains the following columns:
    - id: The id of the sample.
    - context: The context of the statement.
    - statement: The statement to verify.
    - label: The label of the statement (True if factual, False otherwise).
    - explanation: The explanation of the statement.
    - inner_id: The id of the sample in the original dataset.
    - source: Dataset from which the sample comes from.
    - synthetic: Whether the sample is synthetic or not (was generated by a LLM).
    """

    FACTUAL_EXPLANATION = "The statement is factual."

    @staticmethod
    def filter_multiple_sentence_statements(x):
        sentences = nltk.sent_tokenize(x['statement'])
        return len(sentences) == 1

    @staticmethod
    def from_augmented_clinical_notes(path: str, output_path: str, llm_output_column: str = 'output'):
        """
        Constructs the MedHal dataset from a processed augmented clinical notes dataset.

        Args:
            path: Path to the processed augmented clinical notes dataset.
            output_path: Path to the output file where the MedHal samples will be saved.
            llm_output_column: Column name of the LLM output in the augmented clinical notes dataset.
        """
        dataset = HuggingFaceDataset.from_csv(path)
        
        def transform_acm_to_medhal(x):

            assert x['factual'][0] and not x['factual'][1], "The first sample must be correct and the second one must be incorrect"
            assert x['idx'][0] == x['idx'][1], "The first sample must be before the second one"
            assert x['concept'][0] == x['concept'][1], "The first sample must be before the second one"

            if x['concept'][0] == 'sex':
                factual_statement, hallucinated_statement = AugmentedClinicalNotes.fix_sex_generations(x[llm_output_column][0], x[llm_output_column][1])
            else:
                factual_statement = x[llm_output_column][0]
                hallucinated_statement = x[llm_output_column][1]

            if nltk.translate.bleu_score.sentence_bleu([factual_statement.split()], hallucinated_statement.split(), weights=(1, 1, 0, 0)) > 0.8:
                factual_statement = 'None'
                hallucinated_statement = 'None'

            return {
                'id': [str(uuid.uuid4()) for _ in range(len(x['idx']))],
                'context': x['full_note'],
                'statement': [factual_statement, hallucinated_statement],
                'label': [True, False],
                'explanation': [MedHal.FACTUAL_EXPLANATION, factual_statement],
                'inner_id': x['idx'],
                'source': ['acm'] * len(x['idx']),
                'synthetic': [True, False]
            }
        
        dataset = dataset.map(
            transform_acm_to_medhal,
            desc="Transforming ACM to MedHal",
            batched=True,
            batch_size=2,
            remove_columns=dataset.column_names
        )

        dataset = dataset.filter(lambda x: x['statement'] != 'None', desc="Filtering multiple sentence statements")

        return dataset


    @staticmethod
    def from_medmcqa(path: str, output_path: str, llm_output_column: str = 'output'):
        """
        Constructs the MedHal dataset from a processed MedMCQA dataset.

        Args:
            path: Path to the processed MedMCQA dataset.
            output_path: Path to the output file where the MedHal samples will be saved.
            llm_output_column: Column name of the LLM output in the MedMCQA dataset.
        """
        dataset = HuggingFaceDataset.from_csv(path)
        
        def transform_medmcqa_to_medhal(x):

            assert x['is_correct'][0] and not x['is_correct'][1], "The first sample must be correct and the second one must be incorrect"
            assert x['id'][0] == x['id'][1], "The first sample must be before the second one"


            # Determine if context is needed. To do so, we verify if the question has more than one sentence
            # If it does, we remove the last sentence and use it as the context
            # Otherwise, we use 'None' as the context
            if len(nltk.sent_tokenize(x['question'][0])) > 1:
                context = ' '.join(nltk.sent_tokenize(x['question'][0])[:-1])
            else:
                context = 'None'

            return {
                'id': [str(uuid.uuid4()) for _ in range(len(x['id']))],
                'context': [context] * len(x['id']),
                'statement': x[llm_output_column],
                'label': x['is_correct'],
                 # Explanation of the factual statement is the explanation given in the dataset
                 # Explanation of the non-factual statement is the true statement
                'explanation': [x['explanation'][0], x[llm_output_column][0]],
                'inner_id': x['id'],
                'source': ['medmcqa'] * len(x['id']),
                'synthetic': [True] * len(x['id'])
            }
        
        dataset = dataset.map(
            transform_medmcqa_to_medhal, 
            desc="Transforming MedMCQA to MedHal",
            batched=True,
            batch_size=2,
            remove_columns=dataset.column_names
        )

        dataset = dataset.filter(MedHal.filter_multiple_sentence_statements, desc="Filtering multiple sentence statements")

        if output_path is not None:
            dataset.to_csv(output_path, index=False)

        return dataset

    @staticmethod
    def from_medqa(path: str, output_path: str, llm_output_column: str = 'output'):
        """
        Constructs the MedHal dataset from a processed MedQA dataset.

        Args:
            path: Path to the processed MedQA dataset.
            output_path: Path to the output file where the MedHal samples will be saved.
            llm_output_column: Column name of the LLM output in the MedQA dataset.
        """
        dataset = HuggingFaceDataset.from_csv(path)
        
        def transform_medqa_to_medhal(x):
            
            assert x['is_correct'][0] and not x['is_correct'][1], "The first sample must be correct and the second one must be incorrect"
            assert int(x['id'][0]) == int(x['id'][1]) - 1, "The first sample must be before the second one"

            return {
                'id': [str(uuid.uuid4()) for _ in range(len(x['id']))],
                'context': x['context'],
                'statement': x[llm_output_column],
                'label': x['is_correct'],
                'explanation': [MedHal.FACTUAL_EXPLANATION, x[llm_output_column][0]],
                'inner_id': x['id'],
                'source': ['medqa'] * len(x['id']),
                'synthetic': [True] * len(x['id'])
            }

        dataset = dataset.map(
            transform_medqa_to_medhal, 
            desc="Transforming MedQA to MedHal",
            batched=True,
            batch_size=2,
            remove_columns=dataset.column_names
        )
        
        dataset = dataset.filter(MedHal.filter_multiple_sentence_statements, desc="Filtering multiple sentence statements")

        if output_path is not None:
            dataset.to_csv(output_path, index=False)

        return dataset

    @staticmethod
    def from_splitted_sumpubmed(positive_path: str, negative_path: str, output_path: str, llm_output_column: str = 'OUTPUT'):
        """
        Constructs the MedHal dataset from a splitted SumPubMed dataset. The dataset is splitted in the sense
        that positive and negative samples are in different files.

        Args:
            positive_path: Path to the positive samples.
            negative_path: Path to the negative samples.
            output_path: Path to the output file where the MedHal samples will be saved.
            llm_output_column: Column name of the LLM output in the negative samples.
        """
        positive_dataset = HuggingFaceDataset.from_csv(positive_path)
        negative_dataset = HuggingFaceDataset.from_csv(negative_path)

        def generate_positive_samples(x):

            unique_ids = [str(uuid.uuid4()) for _ in range(len(x['text']))]

            return {
                'id': unique_ids,
                'context': x['text'],
                'statement': x['summary'],
                'label': [True] * len(x['text']),
                'explanation': [MedHal.FACTUAL_EXPLANATION] * len(x['text']),
                'inner_id': x['id'],
                'source': ['sumpubmed'] * len(x['text']),
                'synthetic': [False] * len(x['text'])
            }

        positive_ready_dataset = positive_dataset.map(
            generate_positive_samples, 
            remove_columns=positive_dataset.column_names,
            batched=True,
            desc="Generating positive samples"
        )

        def generate_negative_samples(x):
            unique_ids = [str(uuid.uuid4()) for _ in range(len(x['text']))]

            fake_summaries = []
            for before, output, after in zip(x['before'], x[llm_output_column], x['after']):
                fake_summaries.append(f"{before} {output.lower()} {after}".strip())

            explanations = []
            for explanation in x['sentence']:
                explanations.append(f"According to the source document, {explanation}")

            return {
                'id': unique_ids,
                'context': x['text'],
                'statement': fake_summaries,
                'label': [False] * len(x['text']),
                'explanation': explanations,
                'inner_id': x['id'],
                'source': ['sumpubmed'] * len(x['text']),
                'synthetic': [True] * len(x['text'])
            }
        
        print('Before : ', len(negative_dataset))


        negative_dataset = negative_dataset.filter(
            lambda x: 'Here' not in x[llm_output_column] or 'transformed sentence' not in x[llm_output_column],
            desc="Filtering negative samples"
        )

        print(len(negative_dataset))

        negative_ready_dataset = negative_dataset.map(
            generate_negative_samples,
            remove_columns=negative_dataset.column_names,
            batched=True,
            desc="Generating negative samples"
        )

        # Merge the positive and negative datasets
        merged_dataset = concatenate_datasets([positive_ready_dataset, negative_ready_dataset])

        if output_path is not None:
            merged_dataset.to_csv(output_path, index=False)

        return merged_dataset
