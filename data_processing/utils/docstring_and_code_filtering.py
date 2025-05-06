import json
import re

from typing import Dict, Union

def doctring_and_code_filtering(pretokenized_data: Union[str, Dict], save_path: str = None):
    if isinstance(pretokenized_data, str):
        with open("../../parsed_dataset.json", encoding="utf-8") as f:
            full_data = json.load(f)
    else:
        full_data = pretokenized_data

    def tokenize_pretokenized_string(s):
        # Tokenizes strings like [DEF]train[DELIMIT_1_L]... into separate tokens
        return re.findall(r'\[[^\[\]]+\]|[^\[\]]+', s)

    filtered = []
    count_under_1024 = 0
    count_total = 0
    for entry in full_data:
        if "docstring" in entry and "parsed" in entry:
            # Strip control chars, just in case
            doc = entry["docstring"].encode("utf-8", "ignore").decode("utf-8")
            parsed = entry["parsed"].encode("utf-8", "ignore").decode("utf-8")

            # Optional: skip very long entries (based on token count)
            token_count = len(tokenize_pretokenized_string(parsed))
            count_total += 1
            if token_count <= 512:
                count_under_1024 += 1
            if token_count > 512:
                continue

            filtered.append({"docstring": doc, "parsed": parsed})

    print(f"Samples with ≤512 tokens: {count_under_1024} / {count_total} ({(count_under_1024 / count_total) * 100:.2f}%)")

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            for row in filtered:
                json.dump(row, f)
                f.write("\n")

    return filtered
