def preprocess(batch, tokenizer, max_input_length=256, max_output_length=512):
    input_encodings = tokenizer(
        batch["docstring"],
        padding="max_length",
        truncation=True,
        max_length=max_input_length
    )
    target_encodings = tokenizer(
        batch["parsed"],
        padding="max_length",
        truncation=True,
        max_length=max_output_length
    )
    input_encodings["labels"] = target_encodings["input_ids"]
    return input_encodings
