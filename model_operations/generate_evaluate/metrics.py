import json
import evaluate
import numpy as np


def compute_metrics(eval_pred, tokenizer=None, pretokenizer=None):
    """
    Computes BLEU and ROUGE scores given predictions and references.
    """
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    predictions, references = eval_pred

    # predictions and references are already strings
    if pretokenizer:
        predictions = [pretokenizer.reverse(p.strip()) for p in predictions]
        references = [pretokenizer.reverse(r.strip()) for r in references]

    bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in references])
    rouge_result = rouge.compute(predictions=predictions, references=references)

    return {**bleu_result, **rouge_result}


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
