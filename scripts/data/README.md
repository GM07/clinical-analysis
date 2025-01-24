# Data Scripts

This directory contains scripts for data preprocessing and preparation.

## Scripts

### generate_medmcaqa_prompts.py

This script generates prompts from the MedMCQA dataset that will be sent to a larger model to generate samples for the Medical Hallucination Dataset.

Example usage: 

```bash
python scripts/data/generate_medmcaqa_prompts.py --dataset $projects/datasets/medmcqa/ --out $scratch/medmcaqa_prompts --partition 1 --size 20000 --partition_out $scratch/partitions_medmcaqa_prompts/
```
