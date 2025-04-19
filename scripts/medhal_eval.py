from argparse import ArgumentParser
import logging

from src.pipelines.dataset_inference_pipeline import ModelDatasetInferencePipeline

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

from datasets import load_from_disk

from src.data.formatter import Formatter
from src.models.utils import load_tokenizer

parser = ArgumentParser(description='Program that evaluates a model on the medhal dataset')

parser.add_argument('--dataset', type=str, required=True, help='Path to medhal dataset')
parser.add_argument('--model', type=str, required=True, help='Path to model')
parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer')
parser.add_argument('--model_type', type=str, required=True, help='Will indicate how the prompt will be formatted (medhal or general)')
parser.add_argument('--out', type=str, required=True, help='Where the dataset will be saved')

GENERAL_MODEL_TEMPLATE = """### Task Description
- You will evaluate whether a medical statement is factually accurate.
- The statement may reference a provided context.
- Respond with "YES" if the statement is factually correct or "NO" if it contains inaccuracies.
- In order to answer YES, everything in the statement must be supported by the context.
- In order to answer NO, there must be at least one piece of information in the statement that is not supported by the context.
- You must also provide an explanation of why you think the statement is factual or not. If it is factual, put "The statement is factual" as your explanation.
- Your answer should follow the following format :
Factual: [YES/NO]
Explanation: [Your explanation]

### Context
{context}

### Statement
{statement}"""

def prompt_medhal(x, formatter):
    return formatter(x)

# def prompt_general(x):
    # 

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    assert args.model_type in ['medhal', 'general']

    dataset = load_from_disk(args.dataset)
    test = dataset['test']

    if args.model_type == 'medhal':
        tokenizer = load_tokenizer(args.tokenizer)
        formatter = Formatter(tokenizer, training=False)

        test = test.map(
            prompt_medhal, 
            fn_kwargs={
                'formatter':formatter
            }, 
            desc='Formatting dataset'
        )
    # else:
        # pass

    print('Sample : ', test[0]['text'])

    pipeline = ModelDatasetInferencePipeline(args.model, args.tokenizer)
    pipeline(test, apply_chat_template=False, saving_path=args.out, input_column='text')

if __name__ == '__main__':
    main()
