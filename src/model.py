from typing import Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(
    models_path: str, model_dir: str, device="cuda:0"
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load model from local weights.
    """
    model = AutoModelForCausalLM.from_pretrained(f"{models_path}/{model_dir}").to(
        device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(f"{models_path}/{model_dir}")
    return tokenizer, model


def generate_const(tokenizer, model, prompt: str, choices_ids) -> Tuple[str, str]:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        last_token_logits = logits[0, -1, :]  # Logits for the last token
        masked_logits = last_token_logits.clone()
        masked_logits[:] = float("-inf")
        masked_logits[choices_ids] = last_token_logits[choices_ids]
        top_const_token_id = torch.argmax(masked_logits).item()
        top_unconst_token_id = torch.argmax(last_token_logits).item()
        top_const_token = tokenizer.decode([top_const_token_id])
        top_unconst_token = tokenizer.decode([top_unconst_token_id])

    return top_const_token.strip(), top_unconst_token.strip()


from typing import Optional


def generate_unconst(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 64,
    stop_word: Optional[str] = None,
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Exclude prompt
    new_tokens = generated_ids[0][input_ids.size(1) :]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    if stop_word is not None:
        stop_index = new_text.find(stop_word)
        if stop_index != -1:
            new_text = new_text[: stop_index + len(stop_word)]
    return new_text.strip()


class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        self.out, _ = module_outputs


def get_acts(statements, tokenizer, model, layers, device="cuda:0"):
    """
    Get given layer activations for the statements.
    Return dictionary of stacked activations.
    """
    # attach hooks
    hooks, handles = [], []
    for layer in layers:
        hook = Hook()
        handle = model.model.layers[layer].register_forward_hook(hook)
        hooks.append(hook), handles.append(handle)

    # get activations
    acts = {layer: [] for layer in layers}
    for statement in tqdm(statements):
        input_ids = tokenizer.encode(statement, return_tensors="pt").to(device)
        model(input_ids)
        for layer, hook in zip(layers, hooks):
            acts[layer].append(hook.out[0, -1])

    for layer, act in acts.items():
        acts[layer] = torch.stack(act).float()

    # remove hooks
    for handle in handles:
        handle.remove()

    return acts
