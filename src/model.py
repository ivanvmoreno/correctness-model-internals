from typing import Optional, Tuple, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

## If you unconditionally import vllm, it throws errors if you don't have the unix resource module
## Instead, we can just wrap them and call an import vLLM in the load_model function
def maybe_import_vllm():
    try:
        from vllm import LLM, SamplingParams
        from vllm.model_executor.guided_decoding.guided_fields import LLMGuidedOptions
        return LLM, SamplingParams, LLMGuidedOptions
    except ImportError:
        return None, None, None

def load_hf_model(
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


def load_vllm_model(
    models_path: str, model_dir: str, max_model_len=1024
):
    """
    Load vLLM model from local weights.
    """
    LLM, SamplingParams, LLMGuidedOptions = maybe_import_vllm()
    if LLM is None:
        raise ImportError("vLLM is not installed (or not importable on this platform).")    
    tokenizer = AutoTokenizer.from_pretrained(f"{models_path}/{model_dir}")
    model = LLM(
        model=f"{models_path}/{model_dir}",
        max_model_len=max_model_len,
    )
    return tokenizer, model


def generate_const_hf(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompts: List[str],
    choices_ids: List[int],
) -> List[str]:
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
        top_const_tokens = tokenizer.batch_decode(
            top_const_token_ids, skip_special_tokens=True
        )
        top_const_tokens = [t.strip() for t in top_const_tokens]

    return top_const_tokens


def generate_const_vllm(
    model,
    prompts: List[str],
    choices_ids: List[int],
) -> List[str]:
    """
    Generate constrained answers for a batch of prompts (multiple-choice) using vLLM.
    """
    LLM, SamplingParams, LLMGuidedOptions = maybe_import_vllm()
    if LLM is None:
        raise ImportError("vLLM is not installed (or not importable on this platform).")
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
    )
    results = model.generate(
        prompts,
        sampling_params=sampling_params,
        guided_options_request=LLMGuidedOptions(guided_choice=choices_ids),
    )
    generated_tokens = [result.outputs[0].text for result in results]
    return generated_tokens


def generate_unconst_vllm(
    llm,
    prompts: List[str],
    max_new_tokens: int = 1024,
    stop_words: Optional[List[str]] = None,
) -> List[str]:
    """Generate unconstrained answers using vLLM, stopping on any stop word substrings."""
    LLM, SamplingParams, LLMGuidedOptions = maybe_import_vllm()
    if LLM is None:
        raise ImportError("vLLM is not installed (or not importable on this platform).")
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )

    vllm_outputs = llm.generate(prompts, sampling_params)
    generated_texts = [output.outputs[0].text for output in vllm_outputs]

    processed_texts = []
    for text in generated_texts:
        if stop_words:
            earliest_stop = None
            for stop_word in stop_words:
                idx = text.find(stop_word)
                if idx != -1 and (
                    earliest_stop is None or idx < earliest_stop[0]
                ):
                    earliest_stop = (idx, stop_word)
            if earliest_stop is not None:
                text = text[: earliest_stop[0] + len(earliest_stop[1])]
        processed_texts.append(text.strip())

    return processed_texts


def generate_unconst_hf(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompts: List[str],
    max_new_tokens: int = 1024,
    stop_words: Optional[List[str]] = None,
) -> List[str]:
    """Generate unconstrained answers for a batch of prompts (open-ended), stopping on any of a list of stop words if provided."""
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

        if stop_words:
            earliest_stop_index = None
            earliest_stop_word = None
            for sw in stop_words:
                idx = new_text.find(sw)
                if idx != -1 and (
                    earliest_stop_index is None or idx < earliest_stop_index
                ):
                    earliest_stop_index = idx
                    earliest_stop_word = sw
            if earliest_stop_index is not None:
                new_text = new_text[
                    : earliest_stop_index + len(earliest_stop_word)
                ]

        new_texts.append(new_text.strip())

    return new_texts


class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        if isinstance(module_outputs, tuple):
            self.out = module_outputs[0]
        else:
            self.out = module_outputs


def get_transformer_layers(model: AutoModelForCausalLM):
    """
    Return the transformer layers from the model.
    This helper supports models that store their layers under either `model.model.layers` or `model.model.h`.
    """
    if hasattr(model, "model"):
        if hasattr(model.model, "layers"):
            return model.model.layers
        elif hasattr(model.model, "h"):
            return model.model.h
    raise ValueError("Could not locate transformer layers in the given model.")


def get_acts(statements, tokenizer, model, layers, device="cuda:0"):
    """
    Get the given layer activations for all statements processed in one batch.
    This version uses the helper get_transformer_layers to support models that might store
    their transformer blocks in different attributes. It shouldn't change anything if the model was previously compatible.
    """
    hooks = []
    handles = []
    transformer_layers = get_transformer_layers(model)
    for layer in layers:
        hook = Hook()
        handle = transformer_layers[layer].register_forward_hook(hook)
        hooks.append(hook)
        handles.append(handle)

    batch = tokenizer(
        statements, return_tensors="pt", padding=True, truncation=True
    )
    batch = {key: tensor.to(device) for key, tensor in batch.items()}
    model(**batch)