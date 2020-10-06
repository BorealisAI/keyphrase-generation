# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np

from overrides import overrides
from typing import Dict, Optional, List

from allennlp.common.file_utils import cached_path
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.instance import Instance
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.fields import TextField, ArrayField, MetadataField, NamespaceSwappingField
from allennlp.data.dataset_readers import DatasetReader, CopyNetDatasetReader
from keyphrase_generation.utils import read_keyphrase_jsonl_line
from keyphrase_generation.utils.common_utils import KEYPHRASE_SEP_SYMBOL

logger = logging.getLogger(__name__)


@DatasetReader.register("keyphrase_seq2seq_copynet_jsonl_reader")
class KeyphraseDatasetReader(CopyNetDatasetReader):
    """
    Similar to CopyNetDatasetReader but with the capability 
    to read from .jsonl instead of .tsv
    """

    def __init__(self,
                 json_source_field_names: List[str],
                 json_target_field_name: str,
                 target_namespace: str,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 source_max_tokens: Optional[int] = None,
                 target_max_tokens: Optional[int] = None,
                 json_target_field_delimiter: str = ";",
                 lazy: bool = False,
                 ) -> None:
        super().__init__(target_namespace=target_namespace,
                         source_tokenizer=source_tokenizer,
                         target_tokenizer=target_tokenizer,
                         source_token_indexers=source_token_indexers,
                         lazy=lazy)

        self.json_source_field_names = json_source_field_names
        self.json_target_field_name = json_target_field_name
        self.json_target_field_delimiter = json_target_field_delimiter

        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0

    @overrides
    def _read(self, file_path):
        # Reset exceeded counts
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        with open(cached_path(file_path), "r") as data_file:
            logger.info(
                "Reading instances from lines in file at: %s", file_path)

            for line in enumerate(data_file):

                source_sequence, target_sequence = read_keyphrase_jsonl_line(line,
                                                                             self.json_source_field_names,
                                                                             self.json_target_field_name,
                                                                             self.json_target_field_delimiter,
                                                                             KEYPHRASE_SEP_SYMBOL)

                yield self.text_to_instance(source_sequence, target_sequence)

        if self._source_max_tokens and self._source_max_exceeded:
            logger.info(
                "In %d instances, the source token length exceeded the max limit (%d) and were truncated.",
                self._source_max_exceeded,
                self._source_max_tokens,
            )

        if self._target_max_tokens and self._target_max_exceeded:
            logger.info(
                "In %d instances, the target token length exceeded the max limit (%d) and were truncated.",
                self._target_max_exceeded,
                self._target_max_tokens,
            )

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  
        """
        Turn raw source string and target string into an ``Instance``.

        Parameters
        ----------
        source_string : ``str``, required
        target_string : ``str``, optional (default = None)

        Returns
        -------
        Instance
            See the above for a description of the fields that the instance will contain.
        """
        
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
            self._source_max_exceeded += 1
            tokenized_source = tokenized_source[:self._source_max_tokens]
        tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol if there is no match).
        source_to_target_field = NamespaceSwappingField(tokenized_source[1:-1], self._target_namespace)

        meta_fields = {"source_tokens": [x.text for x in tokenized_source[1:-1]]}
        fields_dict = {
                "source_tokens": source_field,
                "source_to_target": source_to_target_field,
        }

        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            if self._target_max_tokens and len(tokenized_target) > self._target_max_tokens:
                self._target_max_exceeded += 1
                tokenized_target = tokenized_target[:self._target_max_tokens]
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)

            fields_dict["target_tokens"] = target_field
            meta_fields["target_tokens"] = [y.text for y in tokenized_target[1:-1]]
            source_and_target_token_ids = self._tokens_to_ids(tokenized_source[1:-1] +
                                                              tokenized_target)
            source_token_ids = source_and_target_token_ids[:len(tokenized_source)-2]
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))
            target_token_ids = source_and_target_token_ids[len(tokenized_source)-2:]
            fields_dict["target_token_ids"] = ArrayField(np.array(target_token_ids))
        else:
            source_token_ids = self._tokens_to_ids(tokenized_source[1:-1])
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))

        fields_dict["metadata"] = MetadataField(meta_fields)

        return Instance(fields_dict)

