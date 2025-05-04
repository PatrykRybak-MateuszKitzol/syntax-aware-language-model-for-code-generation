import os

# === Basic Training Setup ===
MODEL_NAME = "t5-base"
TRAIN_SPLIT_PERCENT = 40
NUM_EPOCHS = 3

# === Model and Tokenizer ===
MODEL_NAME = "t5-base"

# === Input/Output lengths ===
MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 512  # Maximum output length that t5-base supports (93% of your data fits without segmentation)

# === Generation Hyperparameters ===
GENERATION_ARGS = {
    "max_length": MAX_OUTPUT_LENGTH,
    "num_beams": 1,  # You changed this from default
    "no_repeat_ngram_size": 0,  # You added this
    "early_stopping": True,  # You added this
    "length_penalty": 1,
}

# === Output directories ===
EXPERIMENT_DIR = f"outputs/{MODEL_NAME}_split{TRAIN_SPLIT_PERCENT}_epochs{NUM_EPOCHS}"
CHECKPOINTS_DIR = os.path.join(EXPERIMENT_DIR, "checkpoints")
GENERATION_NAME_PART = f"beams{GENERATION_ARGS['num_beams']}_norep{GENERATION_ARGS['no_repeat_ngram_size']}_lenpen{GENERATION_ARGS['length_penalty']}"
GENERATED_OUTPUTS_DIR = os.path.join(EXPERIMENT_DIR, f"generated-outputs_{GENERATION_NAME_PART}")
SAVE_OUTPUTS_PATH = os.path.join(GENERATED_OUTPUTS_DIR, "generated_outputs.json")
FINETUNED_MODEL_DIR = CHECKPOINTS_DIR

# === W&B (Weights & Biases) project tracking ===
PROJECT_NAME = "syntax-aware-language-model-for-code-generation"
RUN_NAME = f"{MODEL_NAME}-split{TRAIN_SPLIT_PERCENT}-epochs{NUM_EPOCHS}-doc2code-run"

# === Dataset path ===
DATASET_PATH = "docstring_and_code.jsonl"

# === Default random seed ===
SEED = 42

# === Training Hyperparameters ===
TRAINING_ARGS = {
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "output_dir": CHECKPOINTS_DIR,
    "num_train_epochs": NUM_EPOCHS,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "eval_steps": 500,
    "save_steps": 500,
    "save_total_limit": 2,
    "logging_steps": 100,
    "bf16": True,
    "report_to": "wandb",
    "run_name": RUN_NAME,
}

# === Generation Settings ===
NUM_EXAMPLES_TO_GENERATE = 500  # Number of examples to use from test set
CHUNK_SIZE = 5  # How many examples to generate at once
RUN_LOGITS_PROCESSOR = True # Whether to use the logits processor (SemanticCodeLogitsMask)

# === Trainig methode settings ===
RUN_SEGEMENTATOR = False
RUN_CUSTOM_LOSS = True
