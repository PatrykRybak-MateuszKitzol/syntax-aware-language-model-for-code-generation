import wandb
import evaluate
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
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



# === CONFIG ===
MODEL_NAME = "t5-base"
MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 512 #Maximum output length that t5-base supports (93% of data goes there without segmentation)
PROJECT_NAME = "syntax-aware-language-model-for-code-generation"
RUN_NAME = "t5-base-doc2code-run-3-10%-of-training-data"



def log_gpu(threshold_warning_gb=4.0):
    print("-" * 30)  # Separator for clarity
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
                print(f"⚠️  WARNING: Low available VRAM ({free_mem_gb:.2f} GiB) — evaluation might crash.")

        except Exception as e:
            print(f"Could not get GPU memory info. Error: {e}")

    else:
        print("CUDA not available, cannot check GPU memory.")

    # System RAM check
    mem = psutil.virtual_memory()
    print(f"System RAM usage: {mem.percent:.2f}% ({mem.used / (1024 ** 3):.2f} GiB / {mem.total / (1024 ** 3):.2f} GiB)")
    print("-" * 30)  # End separator



def evaluate_in_chunks(trainer, dataset, chunk_size=10, save_outputs_path=None, tokenizer=None, raw_dataset=None):
    all_metrics = []
    all_outputs = []
    for start_idx in range(0, len(dataset), chunk_size):
        end_idx = start_idx + chunk_size
        print(f"Evaluating examples {start_idx} to {end_idx - 1}")
        chunk = dataset.select(range(start_idx, min(end_idx, len(dataset))))
        metrics = trainer.evaluate(eval_dataset=chunk)
        all_metrics.append(metrics)

        # Generate outputs if requested
        if save_outputs_path and tokenizer:
            raw_chunk = raw_dataset.select(range(start_idx, min(end_idx, len(dataset)))) if raw_dataset else None
            inputs = raw_chunk["docstring"] if raw_chunk else [""] * len(chunk)
            references = raw_chunk["parsed"] if raw_chunk else [""] * len(chunk)
            model_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True,
                                     max_length=MAX_INPUT_LENGTH).to(trainer.model.device)
            outputs = trainer.model.generate(**model_inputs, max_length=MAX_OUTPUT_LENGTH)
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for i in range(len(inputs)):
                all_outputs.append({
                    "input": inputs[i],
                    "reference": references[i],
                    "prediction": decoded_preds[i]
                })

        torch.cuda.empty_cache()
        gc.collect()
        log_gpu()

    if save_outputs_path and all_outputs:
        with open(save_outputs_path, "w") as f:
            json.dump(all_outputs, f, indent=2)
        print(f"Saved generated outputs to {save_outputs_path}")

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



    # === DATA PREPARATION & MODEL LOADING ===
    dataset = load_dataset("json", data_files="docstring_and_code.jsonl", split="train[:3%]")
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid_split = split_dataset["test"].train_test_split(test_size=0.5, seed=42)
    dataset_dict = {
        "train": split_dataset["train"],
        "validation": test_valid_split["train"],
        "test": test_valid_split["test"]
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)



    # === MAPPING DOCSTRING TO CODE ===
    def preprocess(batch):
        input_enc = tokenizer(
            batch["docstring"],
            padding="max_length",
            truncation=True,
            max_length=MAX_INPUT_LENGTH
        )
        target_enc = tokenizer(
            batch["parsed"],
            padding="max_length",
            truncation=True,
            max_length=MAX_OUTPUT_LENGTH
        )
        input_enc["labels"] = target_enc["input_ids"]
        return input_enc

    tokenized_dataset = {k: v.map(preprocess, batched=True, remove_columns=v.column_names) for k, v in dataset_dict.items()}



    # === EVALUATION METRIC ===
    def compute_metrics(eval_pred):
        bleu = evaluate.load("bleu")
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predicted_ids = np.argmax(predictions, axis=-1)

        decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        print(decoded_preds)
        print(decoded_labels)
        print([[label] for label in decoded_labels])

        return bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])



    # === TRAINING ===
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model) #Each batch is padded only up to the longest sample in that batch

    training_args = TrainingArguments(
        per_device_eval_batch_size=4,
        eval_accumulation_steps=32,
        output_dir="./t5-base-doc2code-checkpoints",
        per_device_train_batch_size=8, # currently the best combo I found so far for RTX5070Ti
        gradient_accumulation_steps=1, # currently the best combo I found so far for RTX5070Ti
        num_train_epochs=1, #later on to be changed to 3 - now is 1 because it's 3 times faster
        learning_rate=5e-5, #common default for T5
        weight_decay=0.01, #L2 regularization
        eval_steps=500, #validation after each 500 steps
        save_steps=500, #checkpoint every 500 steps
        logging_steps=100, #logging loss and learning rata each 100 steps
        save_total_limit=2, #only last 2 checkpoints saved during training
        fp16=True, #16-bit floating point numbers (FP16) instead of (FP32) - 50% less VRAM
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

    # Cleaning after training
    del tokenized_dataset["train"]
    del tokenized_dataset["validation"]
    gc.collect()
    torch.cuda.empty_cache()
    log_gpu()

    raw_subset = dataset_dict["test"].select(range(30))
    tokenized_subset = raw_subset.map(preprocess, batched=True, remove_columns=raw_subset.column_names)

    # === EVALUATION OF FINE-TUNED MODEL===
    print("\n=== Finetuned model evaluation in chunks ===")
    finetuned_metrics = evaluate_in_chunks(trainer, tokenized_subset, chunk_size=10, save_outputs_path="finetuned_outputs.json", tokenizer=tokenizer, raw_dataset=raw_subset)
    avg_finetuned_metrics = average_metrics(finetuned_metrics)
    print("Average Finetuned Metrics:", avg_finetuned_metrics)
    save_metrics_to_file(avg_finetuned_metrics, "finetuned_metrics.json")
    log_gpu()

    # Cleaning after evaluation
    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    log_gpu()



    # === EVALUATION OF BASELINE MODEL===
    baseline_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    baseline_eval_args = TrainingArguments(
        output_dir="./baseline_eval_temp",  # Temporary directory for evaluation outputs (if any)
        per_device_eval_batch_size=1,  # Keep evaluation batch size consistent
        fp16=True,  # Keep fp16 if needed for memory/speed consistency
    )

    baseline_trainer = Trainer(
        model=baseline_model,
        args=baseline_eval_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    print("\n=== Baseline model evaluation in chunks ===")
    baseline_metrics = evaluate_in_chunks(baseline_trainer, tokenized_subset, chunk_size=10, save_outputs_path="baseline_outputs.json", tokenizer=tokenizer, raw_dataset=raw_subset)
    avg_baseline_metrics = average_metrics(baseline_metrics)
    print("Average Baseline Metrics:", avg_baseline_metrics)
    save_metrics_to_file(avg_baseline_metrics, "baseline_metrics.json")
    log_gpu()



    wandb.finish()




if __name__ == "__main__":
    main()