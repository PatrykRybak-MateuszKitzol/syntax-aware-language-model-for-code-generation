from collections import deque
from typing import List, Tuple
import sys
import os
import ast
import json
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PRETOKENIZER_DIR = os.path.join(ROOT_DIR, 'Data', 'pretokenizer')
sys.path.append(PRETOKENIZER_DIR)

# Now import your module
from pretokenizer import pretokenize  # or: from pretokenizer import SomeFunction


def segment_tokens(tokens: List[str], max_len: int, protected_spans: List[Tuple[int, int]]):
    """
    Segments a list of tokens into chunks of at most max_len, minimizing cuts across protected spans.

    Args:
        tokens: A list of token strings.
        max_len: Max number of tokens per segment.
        protected_spans: List of (start, end) index ranges for logical code units (e.g., functions).

    Returns:
        A list of (start, end) indices indicating segment boundaries.
    """
    n = len(tokens)
    cost = [0] * n

    # Step 1: Build cost array
    for l, r in protected_spans:
        for i in range(l, r):
            if 0 <= i < n:
                cost[i] += 1

    # Step 2: Dynamic programming (naive version)
    m = (n + max_len - 1) // max_len  # max possible number of segments
    dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
    prev = [[-1] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0

    for k in range(1, m + 1):
        for i in range(1, n + 1):
            for j in range(max(0, i - max_len), i):
                new_cost = dp[k - 1][j] + cost[i - 1]
                #print(k, i, j, new_cost)
                if new_cost < dp[k][i]:
                    dp[k][i] = new_cost
                    prev[k][i] = j

    # Step 3: Backtrack to get best segmentation
    best_k = min(range(1, m + 1), key=lambda t: dp[t][n])
    cuts = []
    i = n
    while best_k > 0:
        j = prev[best_k][i]
        cuts.append(i)
        i = j
        best_k -= 1
    cuts = sorted(cuts)

    # Step 4: Convert to (start, end) segments
    segments = []
    start = 0
    for end in cuts:
        segments.append((start, end))
        start = end

    return segments


def extract_protected_spans(tokens: List[str]) -> List[Tuple[int, int]]:
    spans = []
    stack = []

    # Line-based spans (from one [NEW_LINE] to the next)
    last_newline = -1
    for i, token in enumerate(tokens):
        if token == "[NEW_LINE]":
            if 0 <= last_newline < i - 1:
                spans.append((last_newline, i - 1))
            last_newline = i
    if last_newline < len(tokens) - 1:
        spans.append((last_newline, len(tokens) - 1))

    # Indentation-based spans
    for i, token in enumerate(tokens):
        if token == "[INDENT]":
            stack.append(i)
        elif token == "[DEDENT]":
            if stack:
                start = stack.pop()
                spans.append((start, i + 1))  # include the [DEDENT] token

    spans.sort()
    return spans


def tokenize_pretokenized_string(s):
    # Tokenizes strings like [DEF]train[DELIMIT_1_L]... into separate tokens
    return re.findall(r'\[[^\[\]]+\]|[^\[\]]+', s)

def pretty_print_span(tokens, span):
    start, end = span
    span_tokens = tokens[start:end]
    print(f"\n=== Span {span} ===")
    pretty_print_tokens(span_tokens)

def pretty_print_spans(tokens, spans):
    for span in spans:
        if span[0] == -1:
            span = (0, span[1])  # assume -1 means start from beginning
        pretty_print_span(tokens, span)

def pretty_print_tokens(tokens):
    indent_level = 0
    indent_str = "    "  # 4 spaces
    output = []
    current_line = ""

    def flush_line():
        nonlocal current_line
        if current_line.strip():  # don't add empty lines
            output.append(indent_str * indent_level + current_line.strip())
        current_line = ""

    for token in tokens:
        if token == "[NEW_LINE]":
            flush_line()
            output.append(indent_str * indent_level + "[NEW_LINE]")
        elif token == "[INDENT]":
            flush_line()
            output.append(indent_str * indent_level + "[INDENT]")
            indent_level += 1
        elif token == "[DEDENT]":
            flush_line()
            indent_level = max(indent_level - 1, 0)
            output.append(indent_str * indent_level + "[DEDENT]")
        elif token == "[BLOCK]" or token == "[RETURN]":
            current_line += " " + token if current_line else token
            flush_line()
        else:
            current_line += " " + token if current_line else token

    flush_line()  # Final flush

    print("\n".join(output))



if __name__ == "__main__":
    examples = open("preprocessed_dataset_100.json", "r").read().split("\n\n\n\n")

    codes = []
    with open("preprocessed_dataset_100.json", "r", encoding="utf-8") as f:
        for line in f: 
            obj = json.loads(line)
            if "code" in obj:
                codes.append(obj["code"])

    examples = []

    for code in codes:
        parsed = pretokenize(ast.parse(code), _use_dedent=True, _use_semantics=True)
        tokens = tokenize_pretokenized_string(parsed)
        examples.append(tokens)

    new_spans = extract_protected_spans(examples[0])
    print(pretty_print_tokens(examples[0]))
    print(new_spans)
    print(segment_tokens(examples[0], 60, new_spans))
    pretty_print_spans(examples[0], new_spans)
