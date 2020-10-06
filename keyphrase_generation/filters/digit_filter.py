# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re

from typing import List
from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_filter import WordFilter


@WordFilter.register("digit_filter")
class DigitAndPunctuationFilter(WordFilter):
    """
    Removes words containing digits, and punctuations (apart from "." "," and "-")
    and returns the filtered list

    Parameters
    ----------
    digit_pattern : str
        Words containing digits (matching this regex pattern) will be removed.
    word_pattern: str
        Words containing only alphabets and the above punctuations are kept
    """

    def __init__(self,
                 digit_pattern,
                 word_pattern) -> None:
        self._digit_pattern = re.compile(digit_pattern)
        self._word_pattern = re.compile(word_pattern)

    @overrides
    def filter_words(self, words: List[Token]) -> List[Token]:
        
        filtered_words = [word for word in words
                          if self._word_pattern.match(word.text) and
                          not self._digit_pattern.match(word.text)]
        return filtered_words
