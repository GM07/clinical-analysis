from typing import List
import torch
import gc

def clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()

def run_inference(model, tokenizer, batched_input: List, max_new_tokens=128):
    """
    Runs inference of a model on a set of inputs

    Args:
        model: Model to use for inference
        tokenizer: Tokenizer of the model (used to apply the chat template)
        batched_input: Inputs batched
        max_new_tokens: Max number of tokens to be generated
    """
    messages = list(map(lambda x: [{"role": "user", "content": x}], batched_input))
    encodeds = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", padding=True)

    model_inputs = encodeds.to(model.device)

    generated_ids = model.generate(model_inputs, max_new_tokens=max_new_tokens)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded
