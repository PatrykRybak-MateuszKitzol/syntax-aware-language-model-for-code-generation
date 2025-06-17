from core.pretokenizer import Pretoknizer

class SegmentatorContract:
    """
    SegmentatorContract is a contract for pretokenizers to enforce rules that allow code to be processed by a segmentator.
    UltimateSegmentator is the only one we use, so for simplicity contract for compatibility is provided here, in core direcotry.
    """

    # tags to by implemented by pretokenizers wanted to be compatible
    _required_tags = ["INDENT", "DEDENT", "NEW_LINE"]

    def __init__(self):
        self._validate_devider_compatibility()

    def _validate_devider_compatibility(self):
        if not isinstance(self, Pretoknizer):
            raise TypeError("Segmentator requires a pretokenizer to be a subclass of Pretoknizer.")
        tags = set( k for k in  self.tags.__dict__.keys() if not k.startswith("_") )
        missing = set(self._required_tags) - tags
        if missing:
            raise ValueError(f"Tags: {missing} are needed by segmentator, but not provided by pretokenizer.")

class Segmentator:
    """
    Segmentator abstraction
    """

    def __init__(self, pretokenizer: SegmentatorContract):
        self.tags = pretokenizer.tags