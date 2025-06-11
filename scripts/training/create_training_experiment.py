from argparse import ArgumentParser
import os
import subprocess

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)


parser = ArgumentParser(description='Creates a training experiment on a dataset according to the MedHal format')
parser.add_argument('--experiment_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--nb_epochs', type=int, required=False, default=1)
parser.add_argument('--batch_size', type=int, required=False, default=2)

CONFIG_YAML_FILE = """# Model arguments
model_name_or_path: /tmp/{model_name}
tokenizer_name_or_path: /tmp/{model_name}
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
per_device_eval_batch_size: {batch_size}
per_device_train_batch_size: {batch_size}
save_strategy: "steps"
eval_strategy: "steps"
save_steps: 500
eval_steps: 500
dataset_num_proc: 48
seed: 42"""

FSDP_CONFIG_YAML = """compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'yes'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""

SLURM_SCRIPT = """#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=128000M
#SBATCH --time=1-00:00:00
#SBATCH --output={log_path}/%N-%j.out

SCRIPT=~/clinical-analysis/src/training/scripts/run_sft.py
PATCH_SCRIPT=~/clinical-analysis/scripts/training/patch.py
CONFIG={config_file_path}
FSDP_CONFIG={fsdp_config_file_path}

export PYTHONPATH=$PYTHONPATH:/home/g/gmehenni/clinical-analysis

echo "SLURM_TMPDIR: $SLURM_TMPDIR"
# Add a condition that if slurm_tmpdir is not set, then stop the script
if [ -z "$SLURM_TMPDIR" ]; then
    echo "SLURM_TMPDIR is not set. Stopping the script."
    exit 1
fi

if [ ! -d "$SLURM_TMPDIR/ENV" ]; then
    virtualenv --no-download $SLURM_TMPDIR/ENV
    echo "Created environment"
    source $SLURM_TMPDIR/ENV/bin/activate
    pip install -q --upgrade pip --no-index
    pip install -q --no-index transformers datasets trl accelerate bitsandbytes peft unsloth deepspeed flash-attn
else
    source $SLURM_TMPDIR/ENV/bin/activate
fi



echo "Patching environment with script $PATCH_SCRIPT"
python $PATCH_SCRIPT --env $SLURM_TMPDIR/ENV/

# check if the dataset is already in the local storage
if [ ! -d "$SLURM_TMPDIR/{dataset_name}" ]; then
    echo "Copying dataset to local storage"
    cp -r {dataset_path} $SLURM_TMPDIR/ #copy the dataset to local storage
fi

if [ ! -d "$SLURM_TMPDIR/{model_name}" ]; then
    echo "Copying model to local storage"
    cp -r {model_path} $SLURM_TMPDIR/
fi


echo "Setting environment variables"
export HF_HOME=$SLURM_TMPDIR/

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "Launching script"
echo "PYTHONPATH : $PYTHONPATH"

export CCL_WORKER_COUNT=0
echo "CCL_WORKER_COUNT : $CCL_WORKER_COUNT"
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file $FSDP_CONFIG --num_processes=4 $SCRIPT $CONFIG --torch_dtype=bfloat16 --bnb_4bit_quant_storage=bfloat16
"""


def main():

    args = parser.parse_args()
    experiment_path = os.path.abspath(args.experiment_path)
    model_path = os.path.abspath(args.model_path)
    dataset_path = os.path.abspath(args.dataset_path)
    nb_epochs = args.nb_epochs
    batch_size = args.batch_size


    # Verify args related to the model
    assert os.path.exists(model_path)
    model_name = os.path.split(model_path)[1]
    print('Model name :', model_name)

    assert os.path.exists(dataset_path)
    dataset_name = os.path.split(dataset_path)[1]
    print('Dataset name :', dataset_name)


    os.makedirs(experiment_path)
    output_folder_path = os.path.join(experiment_path, 'outputs')
    print('Output folder path :', output_folder_path)
    os.makedirs(output_folder_path)

    log_folder_path = os.path.join(experiment_path, 'logs')
    print('Log folder path :', log_folder_path)
    os.makedirs(log_folder_path)


    # Create config file
    config_file_path = os.path.join(experiment_path, 'config.yaml')
    print('Config file path :', config_file_path)
    with open(config_file_path, 'w') as f:
        file = CONFIG_YAML_FILE.format(
            model_name=model_name,
            dataset_name=dataset_name,
            output_dir=output_folder_path,
            nb_epochs=nb_epochs,
            batch_size=batch_size
        )
        f.write(file)

    # Create fdsp config file
    fsdp_config_file_path = os.path.join(experiment_path, 'multi_gpu.yaml')
    print('FSDP config file path :', fsdp_config_file_path)
    with open(fsdp_config_file_path, 'w') as f:
        f.write(FSDP_CONFIG_YAML)

    # Create script file
    script_file_path = os.path.join(experiment_path, 'train.sh')
    print('Script file path :', script_file_path)
    with open(script_file_path, 'w') as f:
        file = SLURM_SCRIPT.format(
            config_file_path=config_file_path,
            fsdp_config_file_path=fsdp_config_file_path,
            log_path=log_folder_path,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            model_name=model_name,
            model_path=model_path
        )
        f.write(file)
    
    result = subprocess.run(['chmod', 'u+x', script_file_path], check=True, text=True, capture_output=True)
    


if __name__ == '__main__':
    main()
