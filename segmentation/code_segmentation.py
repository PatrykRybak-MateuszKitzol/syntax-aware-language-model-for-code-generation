from collections import deque
from typing import List, Tuple, Optional
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
from pretty_printer import pretty_print_span, pretty_print_spans, pretty_print_tokens

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
                # print(k, i, j, new_cost)
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


def tokenize_pretokenized_string(s):
    # Tokenizes strings like [DEF]train[DELIMIT_1_L]... into separate tokens
    return re.findall(r'\[[^\[\]]+\]|[^\[\]]+', s)


def extract_control_structure_span(tokens: List[str], control_tag: str) -> List[Tuple[int, int]]:
    spans = []
    i = 0
    while i < len(tokens):
        if tokens[i] == control_tag:
            header_start = i
            while i < len(tokens) and tokens[i] != "[INDENT]":
                i += 1
            if i < len(tokens) and tokens[i] == "[INDENT]":
                depth = 1
                i += 1
                while i < len(tokens) and depth > 0:
                    if tokens[i] == "[INDENT]":
                        depth += 1
                    elif tokens[i] == "[DEDENT]":
                        depth -= 1
                    i += 1
                spans.append((header_start, i))
        else:
            i += 1
    return spans


def extract_single_line_span(tokens: List[str], keyword_tag: str) -> List[Tuple[int, int]]:
    spans = []
    i = 0
    while i < len(tokens):
        if tokens[i] == keyword_tag:
            start = i
            end = i + 1
            while end < len(tokens) and tokens[end] != "[NEW_LINE]":
                end += 1
            spans.append((start, end))
            i = end
        else:
            i += 1
    return spans


def extract_delimited_spans(tokens: List[str], left_tag: str, right_tag: str) -> List[Tuple[int, int]]:
    spans = []
    stack = []
    for i, token in enumerate(tokens):
        if token == left_tag:
            stack.append(i)
        elif token == right_tag and stack:
            start = stack.pop()
            spans.append((start, i + 1))
    return spans


def extract_protected_spans(
    tokens: List[str],
    tags: Optional[List[str]] = None,
    control_tags: bool = True,
    inline_tags: bool = True,
    delimiters: bool = True
) -> List[Tuple[int, int]]:
    """
    Extract spans of interest from a sequence of tokens based on specific language constructs.

    Args:
        tokens (List[str]): The tokenized input sequence.
        tags (Optional[List[str]]): A list of tags to extract spans for, used when the corresponding flag is False.
        control_tags (bool): If True, extract spans for all known control-structure keywords (e.g., [IF], [FOR], [DEF]).
                             If False, only extract control spans for tags explicitly listed in `tags`.
        inline_tags (bool): If True, extract single-line spans for all known inline keywords (e.g., [RETURN], [RAISE]).
                            If False, only extract inline spans for tags explicitly listed in `tags`.
        delimiters (bool): If True, extract spans enclosed by all known delimiter pairs (e.g., (), [], {}).
                           If False, only extract delimited spans if at least one side of the pair is listed in `tags`.

    Returns:
        List[Tuple[int, int]]: A list of spans, where each span is a tuple of (start_index, end_index).
    """
    spans = []
    stack = []
    tags = tags or []

    if "[NEW_LINE]" in tags:
        last_newline = -1
        for i, token in enumerate(tokens):
            if token == "[NEW_LINE]":
                if 0 <= last_newline < i - 1:
                    spans.append((last_newline, i - 1))
                last_newline = i
        if last_newline < len(tokens) - 1:
            spans.append((last_newline, len(tokens) - 1))

    if "[INDENT]" in tags:
        for i, token in enumerate(tokens):
            if token == "[INDENT]":
                stack.append(i)
            elif token == "[DEDENT]" and stack:
                start = stack.pop()
                spans.append((start, i + 1))

    control_keywords = [
        "[IF]", "[ELIF]", "[ELSE]",
        "[FOR]", "[ASYNC_FOR]",
        "[WHILE]",
        "[DEF]", "[ASYNC_DEF]",
        "[CLASS]",
        "[TRY]", "[EXCEPT]", "[EXCEPT_STAR]", "[FINALLY]",
        "[WITH]", "[ASYNC_WITH]",
        "[MATCH]", "[CASE]", "[MATCH_DEFAULT]", "[MATCH_STAR]"
    ]

    if control_tags:
        for control_tag in control_keywords:
            spans.extend(extract_control_structure_span(tokens, control_tag))
    else:
        for tag in tags:
            if tag in control_keywords:
                spans.extend(extract_control_structure_span(tokens, tag))

    single_line_keywords = [
        "[RAISE]", "[RETURN]", "[YIELD]", "[YIELD_FROM]",
        "[ASSERT]", "[AWAIT]", "[DEL]", "[IMPORT]", "[FROM]",
        "[GLOBAL]", "[NONLOCAL]"
    ]

    if inline_tags:
        for single_tag in single_line_keywords:
            spans.extend(extract_single_line_span(tokens, single_tag))
    else:
        for tag in tags:
            if tag in single_line_keywords:
                spans.extend(extract_single_line_span(tokens, tag))

    delimiter_pairs = [
        ("[DELIMIT_1_L]", "[DELIMIT_1_R]"),
        ("[DELIMIT_2_L]", "[DELIMIT_2_R]"),
        ("[DELIMIT_3_L]", "[DELIMIT_3_R]")
    ]

    if delimiters:
        for left, right in delimiter_pairs:
            spans.extend(extract_delimited_spans(tokens, left, right))
    else:
        for left, right in delimiter_pairs:
            if left in tags or right in tags:
                spans.extend(extract_delimited_spans(tokens, left, right))

    spans.sort()
    return spans


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
        new_tokens = tokenize_pretokenized_string(parsed)
        examples.append(new_tokens)

    new_spans = extract_protected_spans(examples[0], delimiters=True)
    print(pretty_print_tokens(examples[0]))
    print(new_spans)
    # print(segment_tokens(examples[0], 60, new_spans))
    pretty_print_spans(examples[0], new_spans)
