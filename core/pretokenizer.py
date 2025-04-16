"""

"""
import os
import sys

from abc import abstractmethod

class Pretoknizer:
    """
    Pretoknizer abstractcion with enforced rules for segmentator compatibility.
    """

    def __init__(self):
        self.tags = None
        self.tags_symbols = None

    @abstractmethod
    def _set_tags(self):
        """
        Set the tags for the pretokenizer. Needed to fulfill the SegmentatorContract contract.
        Also set the tags_symbols dict for the reverse process.
        """

        pass

    @abstractmethod
    def pretokenize(self):
        """
        Pretokenize the code.
        """

        pass

    @abstractmethod
    def reverse(self):
        """
        Reverse the pretokenization process.
        """

        pass