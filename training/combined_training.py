import wandb
import evaluate
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
import pynvml
import psutil
import json
import gc
import numpy as np

import sys
from pathlib import Path

root = Path().resolve().parent
sys.path.insert(0, str(root))

from pretokenizers.firstpretokenizer import FirstPretokenizer
from training_additions import T5WithModeLoss, LogitsMaskingCallback

# === CONFIG ===
MODEL_NAME = "t5-base"
MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 512
PROJECT_NAME = "syntax-aware-language-model-for-code-generation"
RUN_NAME = "t5-base-doc2code-run-3-10%-of-training-data"

pretokenizer = FirstPretokenizer(_use_dedent=True, _use_semantics=True)

def log_gpu(threshold_warning_gb=4.0):
    print("-" * 30)
    if torch.cuda.is_available():
        try:
            device_index = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_index)
            print(f"Checking memory for GPU: {device_name} (Index: {device_index})")

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mem_gb = meminfo.total / (1024 ** 3)
            free_mem_gb = meminfo.free / (1024 ** 3)

            allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
            peak_allocated = torch.cuda.max_memory_allocated(device_index) / (1024 ** 3)

            print(f"Total VRAM: {total_mem_gb:.2f} GiB")
            print(f"Free VRAM: {free_mem_gb:.2f} GiB")
            print(f"Memory allocated by PyTorch: {allocated:.2f} GiB")
            print(f"Memory reserved by PyTorch:  {reserved:.2f} GiB")
            print(f"Peak memory used (PyTorch):  {peak_allocated:.2f} GiB")

            if free_mem_gb < threshold_warning_gb:
                print(f"\u26a0\ufe0f  WARNING: Low available VRAM ({free_mem_gb:.2f} GiB) — evaluation might crash.")

        except Exception as e:
            print(f"Could not get GPU memory info. Error: {e}")
    else:
        print("CUDA not available, cannot check GPU memory.")

    mem = psutil.virtual_memory()
    print(f"System RAM usage: {mem.percent:.2f}% ({mem.used / (1024 ** 3):.2f} GiB / {mem.total / (1024 ** 3):.2f} GiB)")
    print("-" * 30)

def evaluate_in_chunks(trainer, dataset, chunk_size=10, save_outputs_path=None, tokenizer=None, raw_dataset=None):
    all_metrics = []
    all_outputs = []

    print("[DEBUG] Starting evaluate_in_chunks...")

    for start_idx in range(0, len(dataset), chunk_size):
        end_idx = start_idx + chunk_size
        print(f"Evaluating examples {start_idx} to {end_idx - 1}")

        chunk = dataset.select(range(start_idx, min(end_idx, len(dataset))))
        metrics = trainer.evaluate(eval_dataset=chunk)
        all_metrics.append(metrics)

        if save_outputs_path and tokenizer:
            raw_chunk = raw_dataset.select(range(start_idx, min(end_idx, len(dataset)))) if raw_dataset else None
            inputs = raw_chunk["docstring"] if raw_chunk else [""] * len(chunk)
            references = raw_chunk["parsed"] if raw_chunk else [""] * len(chunk)

            # Tokenize inputs
            model_inputs = tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_LENGTH
            ).to(trainer.model.device)

            # Generate predictions
            outputs = trainer.model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.5,  # ✅ helps stop infinite loops
                eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),  # or semantic_end_id
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                no_repeat_ngram_size=3,
                temperature=1.0
            )

            decoded_preds = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            decoded_preds_raw = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)

            print("✅ [DEBUG] Generation complete. Sample output IDs:")
            print(outputs.sequences[0].tolist())

            for i in range(len(inputs)):
                try:
                    reversed_ref = pretokenizer.reverse(references[i])
                    reversed_pred = pretokenizer.reverse(decoded_preds[i])

                    all_outputs.append({
                        "input": inputs[i],
                        "decoded_input": tokenizer.decode(model_inputs["input_ids"][i], skip_special_tokens=False),
                        "input_token_ids": model_inputs["input_ids"][i].tolist(),
                        "raw_reference": references[i],
                        "raw_prediction": decoded_preds_raw[i],
                        "prediction_token_ids": outputs.sequences[i].tolist(),
                        "reference": reversed_ref,
                        "prediction": reversed_pred,
                        "output_length": len(outputs.sequences[i].tolist())
                    })

                except Exception as e:
                    print(f"❌ Error reversing index {i}: {e}")

        torch.cuda.empty_cache()
        gc.collect()
        log_gpu()

        torch.cuda.empty_cache()
        gc.collect()
        log_gpu()

    if save_outputs_path and all_outputs:
        print(f"[DEBUG] Attempting to save {len(all_outputs)} outputs...")
        try:
            with open(save_outputs_path, "w") as f:
                json.dump(all_outputs, f, indent=2)
            print(f"Saved generated outputs to {save_outputs_path}")
        except Exception as e:
            print(f"❌ Error saving outputs: {e}")

    return all_metrics

def save_metrics_to_file(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {path}")

def average_metrics(metrics_list):
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    averaged = {}
    for key in keys:
        averaged[key] = np.mean([m[key] for m in metrics_list])
    return averaged

def main():
    set_seed(42)
    wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    dataset = load_dataset("json", data_files="docstring_and_code.jsonl", split="train[:15%]")
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid_split = split_dataset["test"].train_test_split(test_size=0.5, seed=42)
    dataset_dict = {
        "train": split_dataset["train"],
        "validation": test_valid_split["train"],
        "test": test_valid_split["test"]
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    semantic_token_ids = [i for i in range(tokenizer.vocab_size) if i not in tokenizer.all_special_ids]
    tags = [v for k, v in pretokenizer.tags.__dict__.items() if not k.startswith("_")]
    tokenizer.add_tokens(tags)
    code_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in tags]
    semantic_start_id = tokenizer.convert_tokens_to_ids(pretokenizer.tags.SEMANTIC_START)
    semantic_end_id = tokenizer.convert_tokens_to_ids(pretokenizer.tags.SEMANTIC_END)
    semantic_token_ids.append(semantic_end_id)
    code_token_ids.remove(semantic_end_id)
    code_token_ids.append(tokenizer.convert_tokens_to_ids("</s>"))

    model = T5WithModeLoss.from_pretrained(
        MODEL_NAME,
        semantic_start_id=semantic_start_id,
        semantic_stop_id=semantic_end_id,
        semantic_token_ids=semantic_token_ids,
        code_token_ids=code_token_ids,
    )
    model.resize_token_embeddings(len(tokenizer))

    print("Custom tags added to tokenizer:")
    for tag in tags:
        tag_id = tokenizer.convert_tokens_to_ids(tag)
        print(f"{tag} -> {tag_id}")

    def preprocess(batch):
        input_enc = tokenizer(
            batch["docstring"], padding="max_length", truncation=True, max_length=MAX_INPUT_LENGTH
        )
        target_enc = tokenizer(
            batch["parsed"], padding="max_length", truncation=True, max_length=MAX_OUTPUT_LENGTH
        )
        input_enc["labels"] = target_enc["input_ids"]
        return input_enc

    tokenized_dataset = {k: v.map(preprocess, batched=True, remove_columns=v.column_names) for k, v in dataset_dict.items()}

    def compute_metrics(eval_pred):
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predicted_ids = np.argmax(predictions, axis=-1)

        decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        print("Raw decoded output:")
        print(decoded_preds[:5])

        decoded_preds = [pretokenizer.reverse(pred.strip()) for pred in decoded_preds]
        decoded_labels = [pretokenizer.reverse(label.strip()) for label in decoded_labels]

        bleu_result = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
        rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

        return {**bleu_result, **rouge_result}

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        per_device_eval_batch_size=4,
        eval_accumulation_steps=32,
        output_dir="./t5-base-doc2code-checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        save_total_limit=2,
        bf16=True,
        report_to="wandb",
        run_name=RUN_NAME,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()

    del tokenized_dataset["train"]
    del tokenized_dataset["validation"]
    gc.collect()
    torch.cuda.empty_cache()
    log_gpu()

    raw_subset = dataset_dict["test"].select(range(5))
    tokenized_subset = raw_subset.map(preprocess, batched=True, remove_columns=raw_subset.column_names)

    print("\n=== Finetuned model evaluation in chunks ===")
    finetuned_metrics = evaluate_in_chunks(trainer, tokenized_subset, chunk_size=5, save_outputs_path="finetuned_outputs.json", tokenizer=tokenizer, raw_dataset=raw_subset)
    avg_finetuned_metrics = average_metrics(finetuned_metrics)
    print("Average Finetuned Metrics:", avg_finetuned_metrics)
    save_metrics_to_file(avg_finetuned_metrics, "finetuned_metrics.json")
    log_gpu()

    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    log_gpu()

    baseline_model = T5WithModeLoss.from_pretrained(
        MODEL_NAME,
        semantic_start_id=semantic_start_id,
        semantic_stop_id=semantic_end_id,
        semantic_token_ids=semantic_token_ids,
        code_token_ids=code_token_ids,
    )
    baseline_model.resize_token_embeddings(len(tokenizer))

    baseline_eval_args = TrainingArguments(
        output_dir="./baseline_eval_temp",
        per_device_eval_batch_size=1,
        bf16=True,
    )

    baseline_trainer = Trainer(
        model=baseline_model,
        args=baseline_eval_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("\n=== Baseline model evaluation in chunks ===")
    baseline_metrics = evaluate_in_chunks(baseline_trainer, tokenized_subset, chunk_size=5, save_outputs_path="baseline_outputs.json", tokenizer=tokenizer, raw_dataset=raw_subset)
    avg_baseline_metrics = average_metrics(baseline_metrics)
    print("Average Baseline Metrics:", avg_baseline_metrics)
    save_metrics_to_file(avg_baseline_metrics, "combined_baseline_metrics.json")
    log_gpu()

    wandb.finish()

if __name__ == "__main__":
    main()
