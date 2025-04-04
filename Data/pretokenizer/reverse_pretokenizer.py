import re
from tags import _Tags, tag_to_symbol

class ReversePretokenizer:

    def __init__(self, _use_dedent=False):
        self._use_dedent = _use_dedent

    def reverse(self, text_with_tags):
        if self._use_dedent:
            return self._reverse_with_dedent(text_with_tags)
        else:
            return self._reverse_without_dedent(text_with_tags)
    
    def _reverse_without_dedent(self, text_with_tags):
        for tag_name, tag in filter(lambda x: not x[0].startswith("_"),  _Tags.__dict__.items()):
            text_with_tags = text_with_tags.replace(tag, tag_to_symbol[tag])
        return text_with_tags

    def _reverse_with_dedent(self, text_with_tags):
        def smart_replace(match):
            tag = match.group(0)
            if tag == _Tags.INDENT:
                smart_replace.indent_level += 1
                return "\n" + tag_to_symbol[_Tags.INDENT] * smart_replace.indent_level
            elif tag == _Tags.DEDENT:
                smart_replace.indent_level -= 1
                return "\n" + tag_to_symbol[_Tags.INDENT] * smart_replace.indent_level
            return tag
        
        def collapse_newlines(text):
            return text

        smart_replace.indent_level = 0
        inter_text = re.sub(fr"{re.escape(_Tags.INDENT)}|{re.escape(_Tags.DEDENT)}", smart_replace, text_with_tags)
        return collapse_newlines(self._reverse_without_dedent(inter_text))

def reverse(text_with_tags, _use_dedent=False):
    reverse_pretokenizer = ReversePretokenizer(_use_dedent)
    return reverse_pretokenizer.reverse(text_with_tags)