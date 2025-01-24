from argparse import ArgumentParser
import logging
import os

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

logger = logging.getLogger(__name__)

parser = ArgumentParser(description='Program that fetches the datasets needed to create the Medical Hallucination Dataset')
parser.add_argument('--out', type=str, required=True, help='Output path where the datasets will be saved')

hf_paths = ['openlifescienceai/medmcqa', 'AGBonnet/augmented-clinical-notes']

def main():
    args = parser.parse_args()

    assert os.path.exists(args.out) and os.path.isdir(args.out), f"Output path {args.out} does not exist and must be a folder"
    
    # Fetch the datasets 
    for path in hf_paths:
        dataset = load_dataset(path, trust_remote_code=True, cache_dir='./cache/')
        dataset.save_to_disk(os.path.join(args.out, path.split('/')[-1]))

if __name__ == '__main__':
    main()

