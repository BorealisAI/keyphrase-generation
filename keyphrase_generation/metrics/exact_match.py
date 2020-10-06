# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

from typing import Set, Dict, List
from overrides import overrides

from allennlp.training.metrics import Metric
from allennlp.common.checks import ConfigurationError


@Metric.register("exact_match_metrics")
class ExactMatchMetrics(Metric):
    """
    Keyphrase evaluation metrics Precision, Recall and F-Score

    Precision = No. of relevant keyphrases generated / No. of total keyphrases generated

    Recall = No. of relevant keyphrases generated / No. of total gold standard relevant keyphrases

    F-Score = 2 * Precision * Recall / Precision + Recall

    Parameters
    ----------
    sep_token: the token (index) which separates the generated text into keyphrases
    exclude_indices: optional (default = None)
        Indices to exclude when evaluating. This should usually include
        the indices of the start, end, and pad tokens.
    top_k: precision, recall and f-score @ K
    """

    def __init__(self,
                 sep_token: int,
                 exclude_indices: Set[int] = None,
                 top_k: int = 5,
                 ) -> None:

        self._sep_token = sep_token
        self._exclude_indices = exclude_indices or set()
        self._top_k = top_k

        self.precision_list = []
        self.recall_list = []
        self.fscore_list = []

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 references: torch.Tensor
                 ):
        """
        Update precision and recall counts

        Parameters
        ----------
        predictions : ``torch.LongTensor``, required
            Batched predicted tokens of shape `(batch_size, max_sequence_length)`.
        references : ``torch.LongTensor``, required
            Batched reference (gold) translations with shape `(batch_size, max_gold_sequence_length)`.
        
        Returns
        -------
        None
        """
        predictions, references = self.unwrap_to_tensors(predictions, references)

        # sanity check
        if references.shape[0] != predictions.shape[0]:
            raise ConfigurationError(
                "references must have same 1st dimension (batch_size) as predictions"
                "found tensor of shape: {}".format(references.size())
            )

        predictions_list = self.get_keyphrase_list(predictions)
        references_list = self.get_keyphrase_list(references)

        for pred, ref in zip(predictions_list, references_list):
            # To compute Precision @ K
            if len(pred) > self._top_k:
                pred = pred[:self._top_k]
            
            p = self.compute_precision(pred, ref)
            self.precision_list.append(p)

            r = self.compute_recall(pred, ref)
            self.recall_list.append(r)

            self.fscore_list.append(self.compute_fscore(p, r))

    def get_keyphrase_list(self, seq_tensor: torch.Tensor) -> List[List[str]]:
        """
        Convert predictions and gold standard into list of lists
        where the inner list contains the K keyphrases 
        corresponding to each item in the batch
        """
        seq_list = []
        seq_tensor = seq_tensor.tolist()

        for seq in seq_tensor:

            # first get rid of "exclude tokens" from the sequence
            seq = [token for token in seq if token not in self._exclude_indices]
            
            keyphrase_list = []
            current_kp = []
            # collate the keyphrases
            for token in seq:
                if token != self._sep_token:
                    current_kp.append(str(token))
                else:
                    keyphrase_list.append(' '.join(current_kp))
                    current_kp = []
            
            keyphrase_list.append(' '.join(current_kp))
            seq_list.append(keyphrase_list)

        return seq_list
        
    @staticmethod
    def compute_precision(pred: List, ref: List) -> float:
        """
        Precision = No. of relevant keywords retrieved / No. of total keywords retrieved
        """
        try:
            precision = len(set(ref).intersection(set(pred))) / len(pred)
            return precision
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def compute_recall(pred: List, ref: List) -> float:
        """
        Recall = No. of relevant keywords retrieved / No. of total relevant keywords
        """

        recall = len(set(ref).intersection(set(pred))) / len(ref)
        return recall

    @staticmethod
    def compute_fscore(precision: float, recall: float) -> float:
        """
        F-Score = 2 * Precision * Recall / Precision + Recall
        """

        numerator = 2 * precision * recall
        denominator = precision + recall

        if numerator == 0:
            return 0
        else:
            return numerator / denominator
    
    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        """
        Returns
        -------
        The accumulated precision, recall and f-score.
        """
        avg_precision = np.round(np.mean(self.precision_list), 4)
        avg_recall = np.round(np.mean(self.recall_list), 4)
        avg_fscore = np.round(np.mean(self.fscore_list), 4)

        if reset:
            self.reset()
        return {"Precision": avg_precision, "Recall": avg_recall, "F-score": avg_fscore}

    @overrides
    def reset(self) -> None:
        """
        Reset accumulators
        """
        self.precision_list = []
        self.recall_list = []
        self.fscore_list = []
