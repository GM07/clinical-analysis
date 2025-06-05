from datasets import DatasetDict, load_from_disk
from src.pipelines.dataset_inference_pipeline import ModelDatasetInferencePipeline

class MedHalValidator:

    PROMPT = """
    ### Task description
    You are tasked to evaluate whether a statement is correctly labeled or not. You will be given a statement and a label indicating whether the label is correct or not. The statement might refer to a provided context. If there is no context, it will be marked as None. 
    
    The label can either be 'Factual' or 'Not factual'. If the label is 'Factual', every information mentioned in the statement must be backed up by general medical knowledge or by an information mentioned in the context. IF the label is 'Not factual', there must exist an information in the statement that is not backed up by general medical knowledge or the context provided.
    
    If no context is provided, the statement will be about general medical knowledge.

    You must answer in the following format without generating any additional text:
    - Answer YES if the label is accurate with what the statement states.
    - Answer NO if the label is not accurate with what the statement states.

    ### Context
    {context}

    ### Statement
    {statement}

    ### Label
    {label}
    """

    def __init__(self, medhal_path: str, model_path: str, tokenizer_path: str = None):
        self.medhal_path = medhal_path
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        
        self.load()

    def load(self):
        self.load_dataset()
        self.load_pipeline()

    def load_pipeline(self):
        self.pipeline = ModelDatasetInferencePipeline(
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path
        )

    def load_dataset(self):
        self.dataset = load_from_disk(self.medhal_path)

    def transform(self) -> DatasetDict:
        def transform_row(row):
            labels = ['Factual' if x else 'Not factual' for x in row['label']]
            contexts = [x if x and len(x) > 0 else 'None' for x in row['context']]
            statements = row['statement']

            return {'prompt': [self.PROMPT.format(
                context=context,
                statement=statement,
                label=label
            ) for label, context, statement in zip(labels, contexts, statements)]}

        return self.dataset.map(transform_row, batched=True)

    def validate(self) -> DatasetDict:
        self.dataset = self.transform()
        print(self.dataset)
        self.dataset['train'] = self.pipeline(self.dataset['train'], input_column='prompt', output_column='output', max_new_tokens=128)
        self.dataset['val'] = self.pipeline(self.dataset['val'], input_column='prompt', output_column='output', max_new_tokens=128)
        self.dataset['test'] = self.pipeline(self.dataset['test'], input_column='prompt', output_column='output', max_new_tokens=128)
        return self.dataset
