import os
from argparse import ArgumentParser

parser = ArgumentParser(description='Pathes the environment to use gloo for training and loads the checkpoint with `weights_only=False` (Assuming Python 3.11)')

parser.add_argument('--env', type=str, required=True, help='Path to environment folder')

PARTIAL_STATE_CODE = f"""if distributed_type is None:
            distributed_type = DistributedType.NO

        return backend, distributed_type"""

PARTIAL_STATE_CODE_UPDATED = f"""if distributed_type is None:
            distributed_type = DistributedType.NO
        backend = 'gloo'
        return backend, distributed_type"""

TRAINER_CODE = f"""checkpoint_rng_state = torch.load(rng_file, weights_only=True)"""
TRAINER_CODE_UPDATED = f"""checkpoint_rng_state = torch.load(rng_file, weights_only=False)"""

def replace_content(path: str, old: str, new: str):

    with open(path, 'r') as f:
        file_content = f.read()

    new_content = file_content.replace(old, new)
    with open(path, 'w') as f:
        f.write(new_content)
    
def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    env = args.env
    path_to_partial_state = os.path.join(env, 'lib/python3.11/site-packages/accelerate/state.py')

    replace_content(path_to_partial_state, PARTIAL_STATE_CODE, PARTIAL_STATE_CODE_UPDATED)

    path_to_trainer = os.path.join(env, 'lib/python3.11/site-packages/transformers/trainer.py')
    replace_content(path_to_trainer, TRAINER_CODE, TRAINER_CODE_UPDATED)

if __name__ == '__main__':
    main()
