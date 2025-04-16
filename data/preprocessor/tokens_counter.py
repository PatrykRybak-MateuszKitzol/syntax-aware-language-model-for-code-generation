from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import re
from tqdm import tqdm

# Very simple code tokenizer (splits on words, numbers, and common symbols)
def simple_tokenize(code):
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\s\w]", code)
    return tokens

# Load the dataset
ds = load_dataset("Nan-Do/code-search-net-python")
train_ds = ds["train"]

token_lengths = []

# Sample subset (you can remove this line to run on full dataset)
# train_ds = train_ds.shuffle(seed=42).select(range(5000))

# Tokenize each code sample
for example in tqdm(train_ds, desc="Tokenizing code"):
    code = example.get("code", "")
    if not code.strip():
        continue
    tokens = simple_tokenize(code)
    token_lengths.append(len(tokens))

# Stats
token_lengths = np.array(token_lengths)
print(f"Total samples: {len(token_lengths)}")
print(f"Mean tokens: {np.mean(token_lengths):.2f}")
print(f"Median tokens: {np.median(token_lengths):.2f}")
print(f"90th percentile: {np.percentile(token_lengths, 90):.2f}")
print(f"95th percentile: {np.percentile(token_lengths, 95):.2f}")
print(f"99th percentile: {np.percentile(token_lengths, 99):.2f}")
print(f"Max tokens: {np.max(token_lengths)}")

# Plot
plt.hist(token_lengths, bins=100, color='lightcoral', edgecolor='black')
plt.title("Simple Token Length Distribution of Python Functions")
plt.xlabel("Number of Simple Tokens")
plt.ylabel("Number of Samples")
plt.grid(True)
plt.show()
