import json

import sys
from pathlib import Path

root = Path().resolve().parent
sys.path.insert(0, str(root))

from utils.pretty_printer import pretty_print_tokens, tokenize_pretokenized_string


def load_predictions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def display_predictions(predictions, title):
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    for i, sample in enumerate(predictions):
        print(f"\n### Example {i + 1}")
        print("\n📌 Input:")
        print(sample["input"])

        print("\n✅ Reference:")
        print(sample["reference"])

        print("\n🤖 Prediction:")
        print(sample["prediction"])

        print("-" * 80)


if __name__ == "__main__":
    baseline = load_predictions("baseline_outputs.json")
    finetuned = load_predictions("finetuned_outputs.json")

    display_predictions(baseline, "Baseline Model Predictions")
    display_predictions(finetuned, "Fine-Tuned Model Predictions")