import json
import ast


def is_valid_python(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        return False
    except Exception as e:
        print(f"Other error: {e}")
        return False

# Load the preprocessed JSON file
with open("preprocessed_dataset.json", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(data)} code examples.")

# Check validity
invalid_snippets = []
for i, item in enumerate(data):
    code = item.get("code", "").strip()
    if not code or not is_valid_python(code):
        invalid_snippets.append((i, code))

print(f"\nInvalid code snippets: {len(invalid_snippets)}")
if invalid_snippets:
    print("Example invalid index:", invalid_snippets[0][0])

for index, code in invalid_snippets:
    print(f"\n--- Invalid snippet at index {index} ---\n{code[:300]}\n")
