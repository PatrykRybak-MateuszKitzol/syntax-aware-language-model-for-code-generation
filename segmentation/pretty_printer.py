import re

def pretty_print_span(tokens, span):
    start, end = span
    span_tokens = tokens[start:(end+1)]
    print(f"\n=== Span {span} ===")
    pretty_print_tokens(span_tokens)


def pretty_print_spans(tokens, spans):
    for span in spans:
        if span[0] == -1:
            span = (0, span[1])  # assume -1 means start from beginning
        pretty_print_span(tokens, span)

def tokenize_pretokenized_string(s):
    # Tokenizes strings like [DEF]train[DELIMIT_1_L]... into separate tokens
    return re.findall(r'\[[^\[\]]+\]|[^\[\]]+', s)

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
