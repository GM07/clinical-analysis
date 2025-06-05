from argparse import ArgumentParser
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Creates a training experiment on a dataset according to the MedHal format')
parser.add_argument('--experiment_path', type=str, required=True)
parser.add_argument('--base_model', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--nb_epochs', type=int, required=False, default=1)

CONFIG_YAML_FILE = """# Model arguments
model_name_or_path: /tmp/{base_model}
tokenizer_name_or_path: /tmp/{base_model}
model_revision: main
torch_dtype: bfloat16
use_flash_attention_2: true

# LoRA arguments
load_in_4bit: true
use_peft: true
lora_r: 16
lora_alpha: 16
lora_target_modules:
- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj

dataset_mixer:
  /tmp/{dataset_name}: 1.0

dataset_splits:
- train
- val

preprocessing_num_workers: 48

# SFT trainer config
bf16: true
do_eval: true
eval_strategy: epoch
gradient_accumulation_steps: 2
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
log_level: info
logging_steps: 5
logging_strategy: steps
max_seq_length: 8192
max_steps: -1
num_train_epochs: {nb_epochs}
output_dir: {output_dir}
overwrite_output_dir: true
per_device_eval_batch_size: 2
per_device_train_batch_size: 2
save_strategy: "steps"
eval_strategy: "steps"
save_steps: 500
eval_steps: 500
dataset_num_proc: 48
seed: 42"""


async def main():

    args = parser.parse_args()

if __name__ == '__main__':
    asyncio.run(main())
