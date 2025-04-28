# === Model and Tokenizer ===
MODEL_NAME = "t5-base"

# === Input/Output lengths ===
MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 512  # Maximum output length that t5-base supports (93% of your data fits without segmentation)

# === Dataset path and split===
DATASET_PATH = "docstring_and_code.jsonl"
TRAIN_SPLIT_PERCENT = 1

# === Default random seed ===
SEED = 42

# === Training Hyperparameters ===
TRAINING_ARGS = {
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "output_dir": "./outputs/checkpoints",
    "num_train_epochs": 1,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "eval_steps": 500,
    "save_steps": 500,
    "save_total_limit": 2,
    "logging_steps": 100,
    "bf16": True,
    "report_to": "wandb",
}

# === W&B (Weights & Biases) project tracking ===
PROJECT_NAME = "syntax-aware-language-model-for-code-generation"
RUN_NAME = f"{MODEL_NAME}-split{TRAIN_SPLIT_PERCENT}-epochs{TRAINING_ARGS['num_train_epochs']}-doc2code-run"
TRAINING_ARGS["run_name"] = RUN_NAME

# === Generation Hyperparameters ===
GENERATION_ARGS = {
    "max_length": MAX_OUTPUT_LENGTH,
    "num_beams": 4,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "length_penalty": 0.7,
}

# === Output directories ===
FINETUNED_MODEL_DIR = f"outputs/{MODEL_NAME}_split{TRAIN_SPLIT_PERCENT}_epochs{TRAINING_ARGS['num_train_epochs']}"
BASELINE_MODEL_DIR = "outputs/baseline_model"  # (optional if you save baseline model separately)
GENERATED_OUTPUTS_DIR = f"{FINETUNED_MODEL_DIR}_generated"

# === Generation Settings ===
NUM_EXAMPLES_TO_GENERATE = 5  # Number of examples to use from test set
CHUNK_SIZE = 1  # How many examples to generate at once
SAVE_OUTPUTS_PATH = f"{GENERATED_OUTPUTS_DIR}/generated_outputs.json"
