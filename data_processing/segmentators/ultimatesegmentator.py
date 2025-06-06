import sys
import os
import re

from typing import List, Tuple, Optional
from random import shuffle
from core.segmentator import Segmentator, SegmentatorContract

from data_processing.pretokenizers.firstpretokenizer import FirstPretokenizer
from data_processing.utils.pretty_printer import pretty_print_span, pretty_print_spans, pretty_print_tokens
from datasets import Dataset

class UltimateSegmentator(Segmentator):

    def __inti__(self, pretokenizer: SegmentatorContract):
        super().__init__(pretokenizer)

    def segment_tokens(self, tokens: List[str], max_len: int, protected_spans: List[Tuple[int, int]]):
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

    def tokenize_pretokenized_string(self, s):
        # Tokenizes strings like [DEF]train[DELIMIT_1_L]... into separate tokens
        return re.findall(r'\[[^\[\]]+\]|[^\[\]]+', s)

    def extract_control_structure_span(self, tokens: List[str], control_tag: str) -> List[Tuple[int, int]]:
        spans = []
        i = 0
        while i < len(tokens):
            if tokens[i] == control_tag:
                header_start = i
                while i < len(tokens) and tokens[i] != self.tags.INDENT:
                    i += 1
                if i < len(tokens) and tokens[i] == self.tags.INDENT:
                    depth = 1
                    block_start = i + 1
                    i += 1
                    while i < len(tokens) and depth > 0:
                        if tokens[i] == self.tags.INDENT:
                            depth += 1
                        elif tokens[i] == self.tags.DEDENT:
                            depth -= 1
                        i += 1
                    spans.append((header_start, i - 1))
            else:
                i += 1
        return spans

    def extract_single_line_span(self, tokens: List[str], keyword_tag: str) -> List[Tuple[int, int]]:
        spans = []
        i = 0
        while i < len(tokens):
            if tokens[i] == keyword_tag:
                start = i
                end = i + 1
                while end < len(tokens) and tokens[end] not in (self.tags.NEW_LINE, self.tags.INDENT, self.tags.DEDENT):
                    end += 1
                spans.append((start, end - 1 if end < len(tokens) and tokens[end] in (
                    self.tags.NEW_LINE, self.tags.INDENT, self.tags.DEDENT) else end))
                i = end
            else:
                i += 1
        return spans

    def extract_delimited_spans(self, tokens: List[str], left_tag: str, right_tag: str) -> List[Tuple[int, int]]:
        spans = []
        stack = []
        for i, token in enumerate(tokens):
            if token == left_tag:
                stack.append(i)
            elif token == right_tag and stack:
                start = stack.pop()
                spans.append((start, i))
        return spans

    def extract_protected_spans(
            self,
            tokens: List[str],
            tags: Optional[List[str]] = None,
            control_tags: bool = False,
            inline_tags: bool = False,
            delimiters: bool = False,
            lines: bool = False,
            indented_blocks: bool = False,
            semantic: bool = False,
            all_options: bool = False,
            strict: bool = False
    ) -> List[Tuple[int, int]]:
        """
        Extract spans of interest from a sequence of tokens based on specific language constructs.

        Args:
            tokens (List[str]): The tokenized input sequence.
            tags (Optional[List[str]]): A list of tags to extract spans for, used when corresponding flags are False.
            control_tags (bool): If True, extract spans for known control-structure keywords.
            inline_tags (bool): If True, extract single-line spans for known inline keywords.
            delimiters (bool): If True, extract spans enclosed by known delimiter pairs.
            lines (bool): If True, extract line-based spans from [NEW_LINE], [INDENT], or [DEDENT] markers.
            indented_blocks (bool): If True, extract indented code blocks delimited by [INDENT] and [DEDENT].
            all_options (bool): If True, enables all other flags (control_tags, inline_tags, delimiters, lines, indented_blocks).
            semantic (bool): If True, extract spans enclosed by [SEMANTIC_START] and [SEMANTIC_END] tags.
            strict (bool): If False (default), filters out spans that are nearly duplicates of others, such as spans that start or end one token apart. (control_tags, inline_tags, delimiters, lines, indented_blocks).

        Returns:
            List[Tuple[int, int]]: A list of unique spans, where each span is a tuple of (start_index, end_index).
        """
        spans = []
        stack = []
        tags = tags or []

        if all_options:
            control_tags = inline_tags = delimiters = lines = indented_blocks = True

        if lines or self.tags.NEW_LINE in tags:
            last_line_break = -1
            i = 0
            while i < len(tokens):
                if tokens[i] in (self.tags.NEW_LINE, self.tags.INDENT, self.tags.DEDENT):
                    if tokens[i] == self.tags.DEDENT:
                        dedent_start = i
                        while i + 1 < len(tokens) and tokens[i + 1] == self.tags.DEDENT:
                            i += 1
                        if 0 <= last_line_break < dedent_start - 1:
                            spans.append((last_line_break, dedent_start - 1))
                        last_line_break = dedent_start
                    else:
                        if 0 <= last_line_break < i - 1:
                            spans.append((last_line_break, i - 1))
                        last_line_break = i
                i += 1
            if last_line_break < len(tokens) - 1:
                spans.append((last_line_break, len(tokens) - 1))

        if indented_blocks or self.tags.INDENT in tags:
            for i, token in enumerate(tokens):
                if token == self.tags.INDENT:
                    stack.append(i)
                elif token == self.tags.DEDENT and stack:
                    start = stack.pop()
                    spans.append((start, i))

        control_keywords = list(filter(lambda x: x != "", [
            self.tags.IF, self.tags.ELIF, self.tags.ELSE,
            self.tags.FOR, self.tags.ASYNC_FOR,
            self.tags.WHILE,
            self.tags.DEF, self.tags.ASYNC_DEF,
            self.tags.CLASS,
            self.tags.TRY, self.tags.EXCEPT, self.tags.EXCEPT_STAR, self.tags.FINALLY,
            self.tags.WITH, self.tags.ASYNC_WITH,
            self.tags.MATCH, self.tags.CASE, self.tags.MATCH_DEFAULT, self.tags.MATCH_STAR
        ]))

        if control_tags:
            for control_tag in control_keywords:
                spans.extend(self.extract_control_structure_span(tokens, control_tag))
        else:
            for tag in tags:
                if tag in control_keywords:
                    spans.extend(self.extract_control_structure_span(tokens, tag))

        single_line_keywords = list(filter(lambda x: x != "", [
            self.tags.RAISE, self.tags.RETURN, self.tags.YIELD, self.tags.YIELD_FROM,
            self.tags.ASSERT, self.tags.AWAIT, self.tags.DEL, self.tags.IMPORT, self.tags.FROM,
            self.tags.GLOBAL, self.tags.NONLOCAL
        ]))

        if inline_tags:
            for single_tag in single_line_keywords:
                spans.extend(self.extract_single_line_span(tokens, single_tag))
        else:
            for tag in tags:
                if tag in single_line_keywords:
                    spans.extend(self.extract_single_line_span(tokens, tag))

        delimiter_pairs = list(filter(lambda x: x[0] != "", [
            (self.tags.DELIMIT_1_L, self.tags.DELIMIT_1_R),
            (self.tags.DELIMIT_2_L, self.tags.DELIMIT_2_R),
            (self.tags.DELIMIT_3_L, self.tags.DELIMIT_3_R)
        ]))

        if delimiters:
            for left, right in delimiter_pairs:
                spans.extend(self.extract_delimited_spans(tokens, left, right))
        else:
            for left, right in delimiter_pairs:
                if left in tags or right in tags:
                    spans.extend(self.extract_delimited_spans(tokens, left, right))

        if semantic or self.tags.SEMANTIC_START in tags or self.tags.SEMANTIC_END in tags:
            spans.extend(self.extract_delimited_spans(tokens, self.tags.SEMANTIC_START, self.tags.SEMANTIC_END))

        spans = sorted(set(spans))

        if not strict:
            filtered = []
            seen = set()
            for start, end in spans:
                if (start, end - 1) in seen or \
                        (start, end + 1) in seen or \
                        (start - 1, end) in seen or \
                        (start + 1, end) in seen or \
                        (start - 1, end + 1) in seen or \
                        (start - 1, end + 1) in seen:
                    continue
                filtered.append((start, end))
                seen.add((start, end))
            spans = filtered

        return spans



    def select_mask_spans(self, spans, total_tokens, min_pct=0.20, max_pct=0.5, max_num_spans=3):
        """
        Selects non-overlapping spans from the given list of spans, ensuring that the 
        total number of tokens in the selected spans falls within the specified range 
        (defined by min_pct and max_pct of total_tokens). The selected spans are returned 
        in sorted order.
        Args:
            spans (list of tuple): A list of spans, where each span is represented as a tuple (start, end).
            total_tokens (int): The total number of tokens available.
            min_pct (float, optional): The minimum percentage of total_tokens that the selected spans should cover. Defaults to 0.15.
            max_pct (float, optional): The maximum percentage of total_tokens that the selected spans can cover. Defaults to 0.3.
            max_num_spans (int, optional): The maximum number of spans to select. Defaults to 3.
        Returns:
            list of tuple: A sorted list of non-overlapping spans that meet the specified 
            criteria. If no spans meet the criteria, the first span from the input list 
            is returned.
        """
        def spans_overlap(a, b):
            if a[0] == b[0] or a[1] == b[1]: return True

            if a[0] > b[0]: first, second = b, a
            else: first, second = a, b
                
            return not (first[1] < second[0])
        # print('tuz przed sortem:', spans.__dict__)
        spans = sorted(spans, key=lambda s: s[0])
        shuffle(spans)

        min_tokens = int(total_tokens * min_pct)
        max_tokens = int(total_tokens * max_pct)

        selected = []
        token_count = 0

        for span in spans:
            start, end = span
            span_len = end - start + 1

            if token_count + span_len > max_tokens:
                continue
            if len(selected) >= max_num_spans:
                break
            if any(spans_overlap(span, s) for s in selected):
                continue

            selected.append(span)
            token_count += span_len

            if token_count >= min_tokens:
                break

        if not selected and spans:
            selected = [spans[0]]

        return sorted(selected, key=lambda s: s[0])

    def apply(self, dataset, tokenizer, input_col = "docstring", label_col = "parsed"):
        """
        Applies the segmentator to the dataset.

        Args:
            dataset: The dataset (list) to be processed.

        Returns:
            A dataset with sampels changed
        """
        def map_fn(example):
            tokens = self.tokenize_pretokenized_string(example[label_col])
            spans = self.extract_protected_spans(tokens, all_options=True)
            spans = self.select_mask_spans(spans, len(tokens))
            additional_offset = 0 
            example[label_col] = ""
            for i in range(len(spans)):
                example[input_col] += f"\n{"".join(tokens[:spans[i][0] + additional_offset] + \
                                            [tokenizer.special_tokens_map['additional_special_tokens'][i]] + \
                                            tokens[spans[i][1] + additional_offset:])}"

                example[label_col] += "".join([tokenizer.special_tokens_map['additional_special_tokens'][i]] + tokens[spans[i][0]:spans[i][1] + 1])
                tokens = tokens[:spans[i][0] + additional_offset]+ tokens[spans[i][1] + additional_offset:]

                additional_offset -= spans[i][1] - spans[i][0] # end - start

            example[label_col] += tokenizer.special_tokens_map['additional_special_tokens'][len(spans)]
            return example
        return list(map(map_fn, dataset))
