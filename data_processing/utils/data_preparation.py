from config import USE_CUSTOM_EOS

def preprocess(batch, tokenizer, use_custom_eos=USE_CUSTOM_EOS, max_input_length=256, max_output_length=512):
    input_encodings = tokenizer(
        batch["docstring"],
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
    )
    target_encodings = tokenizer(
        batch["parsed"],
        padding="max_length",
        truncation=True,
        max_length=max_output_length,
        add_special_tokens=not use_custom_eos
    )
    input_encodings["labels"] = target_encodings["input_ids"]
    return input_encodings
