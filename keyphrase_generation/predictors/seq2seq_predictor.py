# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

from overrides import overrides
from typing import List

from nltk.tokenize import word_tokenize
from allennlp.models import Model
from allennlp.data import Instance, DatasetReader
from allennlp.common.util import JsonDict, sanitize
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.predictors import Predictor, Seq2SeqPredictor
from keyphrase_generation.utils.common_utils import KEYPHRASE_SEP_SYMBOL
END_OF_TITLE = "<eos>"


@Predictor.register("seq2seq_attn")
class Seq2SeqAttnPredictor(Seq2SeqPredictor):

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        processed_outputs = []
        for instance, output in zip(instances, outputs):
            processed_outputs.append(self.process(instance, output))

        return processed_outputs

    def predict_instance(self, instance: Instance) -> JsonDict:
        
        output = self._model.forward_on_instance(instance)
        processed_out = self.process(instance, output)

        return processed_out

    def clean_sequence(self, instance: Instance, field_name: str) -> str:

        sequence = ' '.join([token.text for token in instance.fields[field_name].tokens])
        sequence = sequence.replace(START_SYMBOL, '')
        sequence = sequence.replace(END_SYMBOL, '')
        sequence = sequence.strip()

        return sequence

    def process(self, instance: Instance, output: List) -> JsonDict:

        source_sequence = self.clean_sequence(instance, 'source_tokens')
        target_sequence = self.clean_sequence(instance, 'target_tokens')

        # remove the spaces between keyphrases and the delimiter ";"
        delimiter = ' ' + KEYPHRASE_SEP_SYMBOL + ' '
        target_sequence = ';'.join(target_sequence.split(delimiter))

        source_sequence = ' '.join(word_tokenize(source_sequence))
        source_sequence = source_sequence.replace("@ eot @", END_OF_TITLE)

        # Join tokens to form keyphrases
        # Disregard the separator symbol
        if output['predicted_tokens'] and isinstance(output['predicted_tokens'][0], list):
            output['predicted_tokens'] = output['predicted_tokens'][0]

        keyphrases = ' '.join(output['predicted_tokens']).split(delimiter)
        keyphrases = ';'.join(keyphrases)

        processed_out = {
            "source": source_sequence,
            "target": target_sequence,
            "pred": keyphrases
        }

        return sanitize(processed_out)


    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict["source"]

        target = json_dict["target"]
        target = target.replace(';', KEYPHRASE_SEP_SYMBOL)

        return self._dataset_reader.text_to_instance(source, target)

    def save_attns_to_json(self, outputs, save_file_name):
        """
        For saving the attention matrices,
        - we work on a subset dataset, e.g. of size 50
        - predictor is called with batch_size = size of subset data
        - save all instances to save_file_name
        """
        n_lines = len(outputs) # batch-size
        with open(save_file_name, "w") as f:
            for k in range(n_lines):
                save_dict = {
                    "src_tokens": outputs[k]["metadata"]["source_tokens"],
                    "tgt_tokens": outputs[k]["metadata"]["target_tokens"],
                    "pred_tokens": outputs[k]["predicted_tokens"],
                    "pred_indices": outputs[k]["predictions"].tolist()[0],
                    "attn": outputs[k]["copy_log_probs"].tolist()
                }
                f.write(json.dumps(save_dict))
                f.write("\n")
