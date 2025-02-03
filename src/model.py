from typing import Optional, Tuple, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    models_path: str, model_dir: str, device="cuda:0"
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load model from local weights.
    """
    model = AutoModelForCausalLM.from_pretrained(
        f"{models_path}/{model_dir}"
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(f"{models_path}/{model_dir}")
    return tokenizer, model


def generate_const(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompts: List[str],
    choices_ids: List[int],
) -> Tuple[List[str], List[str]]:
    """Generate constrained answers for a batch of prompts (multiple-choice)."""
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        last_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

        masked_logits = torch.full_like(last_token_logits, float("-inf"))
        masked_logits[:, choices_ids] = last_token_logits[:, choices_ids]
        top_const_token_ids = torch.argmax(masked_logits, dim=-1)
        top_unconst_token_ids = torch.argmax(last_token_logits, dim=-1)
        top_const_tokens = tokenizer.batch_decode(
            top_const_token_ids, skip_special_tokens=True
        )
        top_unconst_tokens = tokenizer.batch_decode(
            top_unconst_token_ids, skip_special_tokens=True
        )
        top_const_tokens = [t.strip() for t in top_const_tokens]
        top_unconst_tokens = [t.strip() for t in top_unconst_tokens]

    return top_const_tokens, top_unconst_tokens


def generate_unconst(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompts: List[str],
    max_new_tokens: int = 1024,
    stop_word: Optional[str] = None,
) -> List[str]:
    """Generate unconstrained answers for a batch of prompts (open-ended)."""
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    generated_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_texts = []
    for i in range(len(generated_ids)):
        input_len = input_ids[i].shape[-1]
        new_tokens = generated_ids[i][input_len:]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        if stop_word is not None:
            stop_index = new_text.find(stop_word)
            if stop_index != -1:
                new_text = new_text[: stop_index + len(stop_word)]

        new_texts.append(new_text.strip())

    return new_texts


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
