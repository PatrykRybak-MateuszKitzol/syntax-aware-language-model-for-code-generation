import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

import torch
import gc
import json
from utils.gpu_logger import log_gpu
import os

def evaluate_in_chunks(
    model,
    dataset,
    chunk_size=10,
    save_outputs_path=None,
    tokenizer=None,
    raw_dataset=None,
    pretokenizer=None,
    max_input_length=256,
    generation_args=None,
    logits_processor=None
):
    """
    Evaluate the model in chunks, generating outputs. Expects generation_args to be passed explicitly.
    """
    if generation_args is None:
        raise ValueError("generation_args must be provided to evaluate_in_chunks()")

    all_outputs = []

    model.eval()

    for start_idx in range(0, len(dataset), chunk_size):
        end_idx = min(start_idx + chunk_size, len(dataset))
        print(f"Generating examples {start_idx} to {end_idx - 1}")

        chunk = dataset.select(range(start_idx, end_idx))

        # Get inputs
        inputs = raw_dataset.select(range(start_idx, end_idx))["docstring"] if raw_dataset else [""] * len(chunk)
        references = raw_dataset.select(range(start_idx, end_idx))["parsed"] if raw_dataset else [""] * len(chunk)

        model_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                **generation_args,
                output_scores=True,
                return_dict_in_generate=True,
                logits_processor=logits_processor,
            )
        decoded_preds = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        for i in range(len(inputs)):
            all_outputs.append({
                "input": inputs[i],
                "reference": pretokenizer.reverse(references[i]) if pretokenizer else references[i],
                "prediction": pretokenizer.reverse(decoded_preds[i]) if pretokenizer else decoded_preds[i]
            })

        torch.cuda.empty_cache()
        gc.collect()
        log_gpu()

    # Save outputs
    if save_outputs_path and all_outputs:
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(save_outputs_path), exist_ok=True)

        with open(save_outputs_path, "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved generated outputs to {save_outputs_path}")

    return all_outputs
