import json
import argparse
import os
from re import S
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
import numpy as np

class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        if isinstance(module_outputs, tuple):
            self.out = module_outputs[0]
        else:
            self.out = module_outputs

def load_hf_model(
    model_path: str, device="cuda"
):
    """
    Load model from local weights.
    """
    # quant_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # quantization_config=quant_config,
        device_map="auto"
    )#.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer, model

def load_statements(dataset_path: str):
    dataset = pd.read_csv(dataset_path)
    prompts = dataset["prompt"].tolist()
    return prompts

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

def get_acts(statements, tokenizer, model, layers, device="cuda"):
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
    batch = {key: tensor for key, tensor in batch.items()} #.to(device)
    model(**batch)

    attention_mask = batch["attention_mask"]
    seq_lengths = attention_mask.sum(dim=1) - 1  # shape: (batch_size,)

    acts = {}
    for layer, hook in zip(layers, hooks):
        batch_indices = torch.arange(hook.out.size(0), device=hook.out.device)
        selected_acts = hook.out[batch_indices, seq_lengths]
        acts[layer] = selected_acts.float()

    for handle in handles:
        handle.remove()

    return acts

def load_activations(
    activations_path: str,
    src_device="cuda",
) -> np.ndarray:
    """Load activations from a given file containing a torch.Tensor."""
    activations = torch.load(activations_path)
    if src_device == "cuda":
        activations = activations.cpu()
    return activations.numpy()

def capture_activations(
    model_path: str,
    save_dir: str,
    layers: list,
    batch_size: int = 20,
    generations_path: str = '',
    device: str = "cuda",
) -> None:
    """Format datasets for question-answering tasks by capturing model activations.

    Args:
        config_path (str): Path to configuration file
        model_id (str): Model to use for generation
        layers (list): List of layers to extract activations from
        batch_size (int, optional): Batch size for generation. Defaults to 25.
        device (str, optional): Device to run inference on. Defaults to 'cuda'.
    """

    tokenizer, model = load_hf_model(
        model_path
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # decoder-only model # changed from left
    #model.to(device)

    # Capture activations for all layers
    if layers is None:
        raise ValueError(
            "Please provide a list of layers to extract activations."
        )

    if isinstance(layers, str):
        try:
            layers = json.loads(layers)
        except json.JSONDecodeError:
            layers = [layers]
    layers = [int(layer) for layer in layers]

    
    # Create save dir if it doesn't exist
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir, ignore_errors=True)
    else:
        print(f"Directory {save_dir} does not exist. Creating.")
    os.makedirs(save_dir, exist_ok=True)

    statements = load_statements(generations_path)

    for batch_num, idx in enumerate(
        range(0, len(statements), batch_size)
    ):
        if batch_num % 10 == 0:
            print(f"Processing batch {batch_num}...")
        chunk = statements[idx : idx + batch_size]
        batch_inputs = []
        for s in chunk:
            batch_inputs.append(s)

        with torch.no_grad():
            acts = get_acts(
                batch_inputs,
                tokenizer,
                model,
                layers,
                device=device,
            )
            if idx < 1:
                print(chunk)
                print(acts[12])

            for l in layers:
                save_path = os.path.join(
                    save_dir,
                    f"layer_{l}",
                )
                os.makedirs(save_path, exist_ok=True)
                torch.save(
                    acts[l],
                    os.path.join(
                        save_path, f"activations_batch_{batch_num}_{idx}.pt"
                    ),
                )

            torch.cuda.empty_cache()
    

    # Join distributions

    # batch_files = os.listdir(save_dir)
    # activations_list = []
    # print(batch_files)
    # for batch_fname in batch_files:
    #     activations_batch_path = os.path.join(
    #         save_dir,
    #         batch_fname,
    #     )
    #     print(activations_batch_path)
    #     activations_batch = load_activations(
    #         activations_batch_path,
    #     )
    #     activations_list.append(activations_batch)
    # activations_joined = np.concatenate(
    #     activations_list, axis=0
    # )

    # activations_joined_path = os.path.join(
    #     save_dir,
    #     "activations_joined.npy",
    # )

    # Remove files?


if __name__ == "__main__":

    model_path = "/workspace/arnau/Llama-3.1-8B-Chat"
    save_dir = "/workspace/arnau/truth_analysis/activations/activations_cities_wrong"
    generations_path = "/workspace/arnau/truth_analysis/prompts/cities_false.csv"
    batch_size = 20
    layers = "[2,4,6,8,10,12,14,16,18,20,22,24,26,28]"


    print("Starting activation capture...")

    capture_activations(model_path, save_dir, layers, batch_size, generations_path)

    print("Activation capture completed successfully.")