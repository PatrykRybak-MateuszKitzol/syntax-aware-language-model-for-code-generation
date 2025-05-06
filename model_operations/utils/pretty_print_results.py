import sys
import json
import os

from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.append(str(root))

from config import GENERATED_OUTPUTS_DIR

# === Paths to load (dynamic based on experiment) ===
PREDICTIONS_FILE = os.path.join(GENERATED_OUTPUTS_DIR, "generated_outputs.json")
METRICS_FILE = os.path.join(GENERATED_OUTPUTS_DIR, "generated_metrics.json")

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def display_predictions(predictions, title="Model Predictions", limit=None):
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    for i, sample in enumerate(predictions):
        if limit and i >= limit:
            break
        print(f"\n### Example {i + 1}")
        print("\n📌 Input:")
        print(sample.get("input", "N/A"))

        print("\n✅ Reference:")
        print(sample.get("reference", "N/A"))

        print("\n🤖 Prediction:")
        print(sample.get("prediction", "N/A"))

        print("-" * 80)

def display_metrics(metrics, title="Evaluation Metrics"):
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    for key, value in sorted(metrics.items()):
        if isinstance(value, (float, int)):
            print(f"{key:<30}: {value:.4f}")
        else:
            print(f"{key:<30}: {value}")
    print("=" * 80)


if __name__ == "__main__":
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"❌ Error: Predictions file not found: {PREDICTIONS_FILE}")
        sys.exit(1)

    if not os.path.exists(METRICS_FILE):
        print(f"❌ Warning: Metrics file not found: {METRICS_FILE}")
        metrics = None
    else:
        metrics = load_json(METRICS_FILE)

    predictions = load_json(PREDICTIONS_FILE)

    print(PREDICTIONS_FILE)

    # Display results
    if metrics:
        display_metrics(metrics, title="Fine-Tuned Model Metrics")

    display_predictions(predictions, title="Fine-Tuned Model Predictions", limit=100)  # Limit to first 10 examples
