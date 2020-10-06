# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright (c) 2019-present, Sean Welleck.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
####################################################################################
# Code for unlikelihood loss is adopted from the paper: 
# Neural Text Generation with Unlikelihood Training (https://arxiv.org/pdf/1908.04319.pdf)
# from https://github.com/facebookresearch/unlikelihood_training by Sean Welleck
####################################################################################

import numpy as np
import torch
import json

import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from collections import defaultdict
from typing import Dict, List, Any
from overrides import overrides

from scipy.linalg import toeplitz

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules import Attention
from allennlp.models.encoder_decoders import CopyNetSeq2Seq
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Metric
from allennlp.modules.attention.dot_product_attention import DotProductAttention
from keyphrase_generation.metrics.exact_match import ExactMatchMetrics
from keyphrase_generation.utils.common_utils import KEYPHRASE_SEP_SYMBOL


@Model.register("copynet_seq2seq_attn")
class Seq2SeqAttnCopy(CopyNetSeq2Seq):
    """
    Basic sequence-to-sequence model with attention mechanism
    """

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 beam_size: int,
                 max_decoding_steps: int,
                 sampling_strategy: str = "greedy",
                 prev_context_len: int = 0,
                 tgt_token_unlikelihood_loss_coefficient: float = 0.0,
                 copy_token_unlikelihood_loss_coefficient: float = 0.0,
                 seq_ul_coefficient: float = 0.0,
                 start_fine_tune_iter: int = 1000000,
                 entropy_reg_coefficient: float = 0.0,
                 nsteps_ahead: int = 0,
                 future_loss_coefficient: float = 0.0,
                 future_tgt_ul: bool = False,
                 future_copy_ul: bool = False,
                 target_embedding_dim: int = 30,
                 scheduled_sampling_ratio: float = 0.0,
                 lower_ss_ratio_every: int = 1500,
                 copy_token: str = "@COPY@",
                 source_namespace: str = "source_tokens",
                 target_namespace: str = "target_tokens",
                 tensor_based_metric: Metric = None,
                 token_based_metric: Metric = None,
                 use_exact_match_metrics: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:

        super().__init__(vocab,
                         source_embedder,
                         encoder,
                         attention,
                         beam_size,
                         max_decoding_steps,
                         target_embedding_dim=target_embedding_dim,
                         copy_token=copy_token,
                         source_namespace=source_namespace,
                         target_namespace=target_namespace,
                         tensor_based_metric=tensor_based_metric,
                         token_based_metric=token_based_metric)

        self.sampling_strategy = sampling_strategy
        
        self.prev_context_len = prev_context_len

        self.tgt_token_unlikelihood_loss_coefficient = tgt_token_unlikelihood_loss_coefficient
        self.copy_token_unlikelihood_loss_coefficient = copy_token_unlikelihood_loss_coefficient
        self.entropy_reg_coefficient = entropy_reg_coefficient
        self.seq_ul_coefficient = seq_ul_coefficient

        # If this value is 0.0 (default), this corresponds to teacher forcing, and
        # if it is 1.0, it corresponds to not using target side ground truth labels.
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._train_iter_counter = 0  # will keep track of training iterations
        # every K iterations, the scheduled_sampling_ratio is decremented by 0.1
        self._lower_ss_ratio_every = lower_ss_ratio_every
        self._start_fine_tune_iter = start_fine_tune_iter

        if use_exact_match_metrics:
            self.pad_index = self.vocab.get_token_index(
                self.vocab._padding_token, self._target_namespace)
            self.sep_index = self.vocab.get_token_index(
                KEYPHRASE_SEP_SYMBOL, self._target_namespace)
            self._exact_match_metrics = ExactMatchMetrics(sep_token=self.sep_index,
                                                          exclude_indices={self.pad_index, self._end_index, self._start_index})
        else:
            self._exact_match_metrics = None

        # @future-token-pred
        self.nsteps_ahead = nsteps_ahead
        self.future_loss_coefficient = future_loss_coefficient
        self.future_tgt_ul = future_tgt_ul
        self.future_copy_ul = future_copy_ul
        
        if self.nsteps_ahead > 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Dictionary to store 
            # Linear layers to transform current decoder hidden state
            # For step_i prediction
            self._state_transform_layers = dict()
            for i in range(1, self.nsteps_ahead+1):
                self._state_transform_layers[i] = Linear(self.decoder_output_dim, self.decoder_output_dim)
                self._state_transform_layers[i].to(device)

            # For attention computation -> to obtain context vector -> to be used for n-step ahead prediction
            self._future_attention = DotProductAttention(normalize=False)

    @overrides
    def forward(self,  
                source_tokens: Dict[str, torch.LongTensor],
                source_token_ids: torch.Tensor,
                source_to_target: torch.Tensor,
                metadata: List[Dict[str, Any]],
                target_tokens: Dict[str, torch.LongTensor] = None,
                target_token_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``, required
            The output of `TextField.as_array()` applied on the source `TextField`. This will be
            passed through a `TextFieldEmbedder` and then through an encoder.
        source_token_ids : ``torch.Tensor``, required
            Tensor containing IDs that indicate which source tokens match each other.
            Has shape: `(batch_size, trimmed_source_length)`.
        source_to_target : ``torch.Tensor``, required
            Tensor containing vocab index of each source token with respect to the
            target vocab namespace. Shape: `(batch_size, trimmed_source_length)`.
        metadata : ``List[Dict[str, Any]]``, required
            Metadata field that contains the original source tokens with key 'source_tokens'
            and any other meta fields. When 'target_tokens' is also passed, the metadata
            should also contain the original target tokens with key 'target_tokens'.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
            Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
            target tokens are also represented as a `TextField` which must contain a "tokens"
            key that uses single ids.
        target_token_ids : ``torch.Tensor``, optional (default = None)
            A tensor of shape `(batch_size, target_sequence_length)` which indicates which
            tokens in the target sequence match tokens in the source sequence.

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        state = self._encode(source_tokens)
        state["source_token_ids"] = source_token_ids
        state["source_to_target"] = source_to_target
        
        if target_tokens:

            if self.training:
                self._train_iter_counter += 1

                if self._train_iter_counter % self._lower_ss_ratio_every == 0:
                    self._scheduled_sampling_ratio += 0.1
                    self._scheduled_sampling_ratio = min(
                        self._scheduled_sampling_ratio, 1.0)  # at most can take value 1.0

            state = self._init_decoder_state(state)
            # Computing MLE loss
            output_dict = self._forward_loss(
                target_tokens, target_token_ids, state)

            # Computing unlikelihood loss
            tgt_token_unlikelihood_loss = self._compute_token_level_unlikelihood_loss(output_dict["tgt_log_probs"],
                                                                                      target_tokens)
            output_dict["tgt_token_unlikelihood_loss"] = self.tgt_token_unlikelihood_loss_coefficient * tgt_token_unlikelihood_loss

            copy_token_level_unlikelihood_loss = self._compute_copy_token_level_unlikelihood_loss(output_dict["copy_log_probs"],
                                                                                                  output_dict["target_in_source"])
            output_dict["copy_token_level_unlikelihood_loss"] = self.copy_token_unlikelihood_loss_coefficient * copy_token_level_unlikelihood_loss

            # only after the pre-training stage, that we introduce the seq-level unlikelihood loss
            if self._train_iter_counter > self._start_fine_tune_iter:
                seq_unlikelihood_loss_using_tgt_tokens = self._compute_seq_level_unlikelihood_loss(output_dict["decoder_predictions"], 
                                                                                                   target_tokens,
                                                                                                   output_dict["stepwise_loss"])
                output_dict["seq_unlikelihood_loss"] = self.seq_ul_coefficient * seq_unlikelihood_loss_using_tgt_tokens
            else:
                # else use a zero tensor for seq_unlikelihood_loss
                output_dict["seq_unlikelihood_loss"] = torch.tensor(0.0, dtype=tgt_token_unlikelihood_loss.dtype, requires_grad=False)
                output_dict["seq_unlikelihood_loss"] = output_dict["seq_unlikelihood_loss"].to(tgt_token_unlikelihood_loss.device)
            
            # entropy regularization term : to overcome peaky output distribution issue
            # -ve since we want the entropy to be higher 
            # i.e., higher entropy will result in a lower overall loss value
            entropy_reg = -1 * self.entropy_reg_coefficient * output_dict["entropy"]
            
            # Final loss is sum of the four losses
            output_dict["loss"] = output_dict["mle_loss"] \
                                + output_dict["future_loss"] \
                                + output_dict["tgt_token_unlikelihood_loss"] \
                                + output_dict["copy_token_level_unlikelihood_loss"] \
                                + output_dict["seq_unlikelihood_loss"] \
                                + entropy_reg

        else:
            output_dict = {}

        output_dict["metadata"] = metadata
        
        if not self.training:
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if target_tokens:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                # shape: (batch_size, target_sequence_length)
                gold_tokens = self._gather_extended_gold_tokens(target_tokens["tokens"],
                                                                source_token_ids,
                                                                target_token_ids)

                if self._tensor_based_metric is not None:
                    self._tensor_based_metric(
                        best_predictions, gold_tokens)  # type: ignore

                if self._exact_match_metrics:
                    self._exact_match_metrics(best_predictions, gold_tokens)

                if self._token_based_metric is not None:
                    predicted_tokens = self._get_predicted_tokens(output_dict["predictions"],
                                                                  metadata,
                                                                  n_best=1)
                    self._token_based_metric(predicted_tokens,  # type: ignore
                                             [x["target_tokens"] for x in metadata])

        return output_dict

    @overrides
    def _forward_loss(self,
                      target_tokens: Dict[str, torch.LongTensor],
                      target_token_ids: torch.Tensor,
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate the loss against gold targets.
        """
        batch_size, target_sequence_length = target_tokens["tokens"].size()
        
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        num_decoding_steps = target_sequence_length - 1
        # We use this to fill in the copy token index when the previous input was copied.
        # shape: (batch_size,)
        copy_input_choices = source_mask.new_full(
            (batch_size,), fill_value=self._copy_index)
        # shape: (batch_size, trimmed_source_length)
        copy_mask = source_mask[:, 1:-1].float()
        # We need to keep track of the probabilities assigned to tokens in the source
        # sentence that were copied during the previous timestep, since we use
        # those probabilities as weights when calculating the "selective read".
        # shape: (batch_size, trimmed_source_length)
        selective_weights = state["decoder_hidden"].new_zeros(copy_mask.size())

        # Indicates which tokens in the source sentence match the current target token.
        # shape: (batch_size, trimmed_source_length)
        target_to_source = state["source_token_ids"].new_zeros(
            copy_mask.size())

        # This is just a tensor of ones which we use repeatedly in `self._get_ll_contrib`,
        # so we create it once here to avoid doing it over-and-over.
        generation_scores_mask = state["decoder_hidden"].new_full((batch_size, self._target_vocab_size),
                                                                  fill_value=1.0)

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full(
            (batch_size,), fill_value=self._start_index)

        step_predictions = []
        step_entropy = []
        step_log_likelihoods = []
        stepwise_log_probs_over_tgt_vocab = []
        stepwise_log_probs_over_copy_vocab = []
        stepwise_target_in_source = []

        # @future-token-pred
        future_step_log_likelihoods = {i:[] for i in range(1, self.nsteps_ahead+1)}
        future_stepwise_log_probs_over_tgt_vocab = {i:[] for i in range(1, self.nsteps_ahead+1)}
        future_stepwise_log_probs_over_copy_vocab = {i:[] for i in range(1, self.nsteps_ahead+1)}
        future_stepwise_target_in_source = {i:[] for i in range(1, self.nsteps_ahead+1)}

        for timestep in range(num_decoding_steps):
            if torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at a rate of 1 - _scheduled_sampling_ratio
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = target_tokens["tokens"][:, timestep]
            # If the previous target token was copied, we use the special copy token.
            # But the end target token will always be THE end token, so we know
            # it was not copied.
            if timestep < num_decoding_steps - 1:
                # Get mask tensor indicating which instances were copied.
                # shape: (batch_size,)
                copied = ((input_choices == self._oov_index) &
                          (target_to_source.sum(-1) > 0)).long()
                # shape: (batch_size,)
                input_choices = input_choices * \
                    (1 - copied) + copy_input_choices * copied
                # shape: (batch_size, trimmed_source_length)
                target_to_source = state["source_token_ids"] == target_token_ids[:,timestep+1].unsqueeze(-1)

            # Update the decoder state by taking a step through the RNN.
            state = self._decoder_step(input_choices, selective_weights, state)
            # Get generation scores for each token in the target vocab.
            # shape: (batch_size, target_vocab_size)
            generation_scores = self._get_generation_scores(state)
            
            # shape: (batch_size, num_classes)
            # class_probabilities = F.softmax(generation_scores, dim=-1)
            vocab_distibution = torch.distributions.Categorical(logits=generation_scores)
            # compute stepwise entropy - for entropy regularization loss of the output distribution
            step_entropy.append(vocab_distibution.entropy().unsqueeze(1))

            if self.sampling_strategy == "greedy":
                # greedy decoding
                # shape (predicted_classes): (batch_size,)
                # _, predicted_classes = torch.max(class_probabilities, 1)
                _, predicted_classes = torch.max(generation_scores, 1)
            else:
                # sampling from category distribution
                torch.random.manual_seed(133) # for deterministic sampling # same as allenNLP torch seed
                torch.cuda.manual_seed_all(133)
                predicted_classes = vocab_distibution.sample()

            # Note: we do not consider copyable source tokens in the sampling step
            # i.e., we sample only from the target vocabulary
            # so that we can feed the target embedding to the decoder in the next time step
            # if the predicted token was actually a copied one, then the sampled token id from the tgt vocab may or may not be the UNK token
            last_predictions = predicted_classes
            step_predictions.append(last_predictions.unsqueeze(1))
            # Get copy scores for each token in the source sentence, excluding the start
            # and end tokens.
            # shape: (batch_size, trimmed_source_length)
            copy_scores = self._get_copy_scores(state)
            # shape: (batch_size,)
            step_target_tokens = target_tokens["tokens"][:, timestep + 1]
            
            step_log_likelihood, selective_weights = self._get_ll_contrib(
                generation_scores,
                generation_scores_mask,
                copy_scores,
                step_target_tokens,
                target_to_source,
                copy_mask)
            step_log_likelihoods.append(step_log_likelihood.unsqueeze(1))

            # keep track of probabilities over the target vocab at each time step
            log_probs = util.masked_log_softmax(
                generation_scores, generation_scores_mask)
            stepwise_log_probs_over_tgt_vocab.append(log_probs)

            # keep track of probabilities over the source tokens at each time step
            log_probs = util.masked_log_softmax(copy_scores, copy_mask)
            stepwise_log_probs_over_copy_vocab.append(log_probs)

            # at each time step, we need to keep track of the occurence of ground truth target
            # in the source sequence --> 1 if it refers to the same word, 0 otherwise
            stepwise_target_in_source.append(target_to_source.float())

            # @future-token-pred
            for step_i in range(1, self.nsteps_ahead+1): # 0 is current token prediction, 1 is one token ahead and so on
                if timestep + step_i < num_decoding_steps: 
                    step_target_tokens = target_tokens["tokens"][:, timestep + step_i + 1]
                    target_to_source = state["source_token_ids"] == target_token_ids[:, timestep + step_i + 1].unsqueeze(-1)
                    future_step_log_likelihood, generation_scores, copy_scores = self._compute_future_step_ll(step_i, 
                                                                                                state, 
                                                                                                generation_scores_mask, 
                                                                                                step_target_tokens, 
                                                                                                target_to_source, 
                                                                                                copy_mask)
                    future_step_log_likelihoods[step_i].append(future_step_log_likelihood.unsqueeze(1))

                    # keep track of probabilities over the target vocab at each time step
                    log_probs = util.masked_log_softmax(
                        generation_scores, generation_scores_mask)
                    future_stepwise_log_probs_over_tgt_vocab[step_i].append(log_probs)

                    # keep track of probabilities over the source tokens at each time step
                    log_probs = util.masked_log_softmax(copy_scores, copy_mask)
                    future_stepwise_log_probs_over_copy_vocab[step_i].append(log_probs)

                    # at each time step, we need to keep track of the occurence of ground truth target
                    # in the source sequence --> 1 if it refers to the same word, 0 otherwise
                    future_stepwise_target_in_source[step_i].append(target_to_source.float())
        
        # Gather stepwise log probabilities
        # to get tensor of dim [batch_size x max_tgt_len x tgt_vocab_size]
        stepwise_log_probs_over_tgt_vocab = torch.stack(stepwise_log_probs_over_tgt_vocab, dim=1)
        stepwise_log_probs_over_copy_vocab = torch.stack(stepwise_log_probs_over_copy_vocab, dim=1)
        stepwise_target_in_source = torch.stack(stepwise_target_in_source, dim=1)
        step_predictions = torch.cat(step_predictions, 1)
        step_entropy = torch.cat(step_entropy, 1)

        # Gather step log-likelihoods scores
        # shape: (batch_size, num_decoding_steps = target_sequence_length - 1)
        log_likelihoods = torch.cat(step_log_likelihoods, 1)
        # Get target mask to exclude likelihood contributions from timesteps after
        # the END token.
        # shape: (batch_size, target_sequence_length)
        target_mask = util.get_text_field_mask(target_tokens)
        # The 0th timestep is just the START token, which is not included in the likelihoods.
        # shape: (batch_size, num_decoding_steps)
        target_mask = target_mask[:, 1:].float()
        # Sum of step log-likelihoods.
        log_likelihoods = log_likelihoods * target_mask
        log_likelihood = log_likelihoods.sum(dim=-1)  # shape: (batch_size,)
        # The loss is the negative log-likelihood, averaged over the batch.
        loss = - log_likelihood.sum() / batch_size
        
        
        # @future-token-pred
        if self.nsteps_ahead > 0:
            total_future_loss = self._compute_total_future_losses(target_tokens, 
                                                future_step_log_likelihoods, 
                                                future_stepwise_log_probs_over_tgt_vocab, 
                                                future_stepwise_log_probs_over_copy_vocab, 
                                                future_stepwise_target_in_source)
        else:
            total_future_loss = torch.zeros_like(loss)

        return {"mle_loss": loss,
                "future_loss": total_future_loss,
                "stepwise_loss": log_likelihoods,
                "entropy": step_entropy.mean(), 
                "tgt_log_probs": stepwise_log_probs_over_tgt_vocab,
                "copy_log_probs": stepwise_log_probs_over_copy_vocab,
                "target_in_source": stepwise_target_in_source,
                "decoder_predictions": step_predictions}

    def _compute_token_level_unlikelihood_loss(self, tgt_log_probs, target_tokens, step_i=0):
        """
        Calculate the token level unlikelihood loss
        - At each time step, we look at all the words in the the ground truth from the previous time steps
        - which forms the negative list at that time step
        - loss is calculated by penalizing the probability of predicting these words present the negative candidate list
        """

        # tgt_log_probs is [batch_size x max_tgt_len x tgt_vocab_size]
        # Collapse it to a 2d tensor:
        # to get tensor of dim [(batch_size * max_tgt_len) x tgt_vocab_size]
        batch_size, max_tgt_len, tgt_vocab_size = tgt_log_probs.size()
        tgt_log_probs = tgt_log_probs.view(-1, tgt_log_probs.size(-1))
        
        with torch.no_grad():
            # The first timestep is just the @START@ token, which is not included in the likelihoods
            # i.e., we do not ask the model to predict the @START@ token
            target = target_tokens["tokens"][:, (step_i+1):]
            # Get the context candidates
            ctx_cands = target.unsqueeze(1).expand(
                target.size(0), target.size(1), target.size(1))
            # get the lower triangular matrix
            ctx_cands = (ctx_cands.tril(-1) + self.pad_index)

            # what is the point of these 2 lines (?)
            # -- taken from https://github.com/facebookresearch/unlikelihood_training/blob/master/custom/candidate_penalty_ce_loss.py
            # ctx_cands_ = ctx_cands_ * ctx_cands_.triu()
            # ctx_cands = ctx_cands.tril(-1) + ctx_cands_

            # Don't include the target for that timestep as a negative target
            # i.e., remove it if it was a part of the candidate list
            ctx_cands = ctx_cands.masked_fill(
                ctx_cands == target.unsqueeze(2), self.pad_index)

            if self.prev_context_len > 0:
                # to consider only a pre-specified previous context size
                # len(mask_generator) == max_tgt_len
                mask_generator = [0] + [1]*self.prev_context_len + \
                    [0]*(max_tgt_len-self.prev_context_len-1)
                # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.toeplitz.html
                mask = toeplitz(mask_generator, np.zeros_like(mask_generator))
                # with the above mask, only the prev n words in the context are considered
                mask = torch.tensor(mask, dtype=ctx_cands.dtype,
                                    requires_grad=False)
                mask = mask.to(ctx_cands.device)
                # create the same mask for the entire batch
                mask = mask.unsqueeze(0).expand(
                    batch_size, max_tgt_len, max_tgt_len)

                # incorporate the previous-N-context-only mask
                ctx_cands = mask * ctx_cands

            # reshape to [(batch_size * max_tgt_len), max_tgt_len]
            ctx_cands = ctx_cands.view(-1, ctx_cands.size(-1))

            # get a zero matrix of the same size size as tgt_log_probs ---> [(batch_size * max_tgt_len) x tgt_vocab_size]
            # for each row, fill 1s in the positions that correspond to the candidate token ids
            # negative_targets ---> is a k-hot vector with the token idx of the candidates as 1
            negative_targets = torch.zeros_like(
                tgt_log_probs).scatter_(1, ctx_cands, 1)

        # - compute loss
        # tgt_log_probs refer to log of probabilities, exp() of it gives the actual prob. values
        # [(batch_size * max_tgt_len) x tgt_vocab_size]
        one_minus_probs = torch.clamp((1.0 - tgt_log_probs.exp()), min=1e-5)

        # only keep the probabilities at the negative token indices
        # [(batch_size * max_tgt_len) x tgt_vocab_size]
        loss = -torch.log(one_minus_probs)*negative_targets
    
        loss = loss.sum(-1)/negative_targets.sum(-1)
        loss = loss.reshape(batch_size, max_tgt_len)
        loss = loss.sum(-1).mean() # average across the batch

        return loss

    def _compute_copy_token_level_unlikelihood_loss(self, copy_log_probs, target_in_source):
        """
        Calculate the copy token level unlikelihood loss
        - at each time step, we look at all the previous occurences of the target (context words) in the source 
        - which forms the negative list at that time step
        - loss is calculated by penalizing the copying probability assiged to those source tokens

        copy_log_probs: [batch_size, max_tgt_len x copy_vocab_size]
        target_in_source: [batch_size, max_tgt_len x copy_vocab_size]
        """

        batch_size, max_tgt_len, copy_vocab_size = copy_log_probs.size()

        with torch.no_grad():
            # candidate list
            # cumsum will accumulate the copy tokens from the source across timesteps
            copy_cands = torch.cumsum(target_in_source, 1)
            # since sum will result in values > 1, we only want to use this only as a mask of 0s and 1s
            copy_cands = (copy_cands >= 1).float()

            # don't include a source token in the negative candidate list
            # if the gold standard at that time step is in the copy token blacklist
            # done by subtracting out the target_in_source at that time step from the accumulated copy candidates
            # check notes for example
            negative_copy_mask = copy_cands - target_in_source
            # 1 if that idx is in the negative list and its probability should be considered for copy level unlikelihood loss
            # 0 otherwise

        # - compute loss
        # lprobs refer to log of probabilities, exp() of it gives the actual prob. values
        # [batch_size x max_tgt_len x copy_vocab_size]
        one_minus_probs = torch.clamp((1.0 - copy_log_probs.exp()), min=1e-5)

        # only keep the probabilities at the negative token indices
        # [batch_size x max_tgt_len x copy_vocab_size]
        loss = -torch.log(one_minus_probs)*negative_copy_mask
        negative_copy_mask = negative_copy_mask.sum(-1) + 1.0 # adding 1 to prevent div by zero error
        loss = loss.sum(dim=-1)
        loss = loss / negative_copy_mask

        loss = loss.sum(-1).mean() # sum across the sequence and then mean across the batch

        return loss

    def _compute_seq_level_unlikelihood_loss(self, pred_tokens, target_tokens, stepwise_tgt_logprobs):
        """
        To mask out the repeated tokens in the GENERATED text
        """
        # only consider the non pad timesteps
        target = target_tokens["tokens"][:, 1:]
        pad_tokens_mask = (target == self.pad_index)==0 # corresponds to non-pad tokens
        pad_tokens_mask = torch.tensor(pad_tokens_mask, dtype=stepwise_tgt_logprobs.dtype, requires_grad=False)
        pad_tokens_mask = pad_tokens_mask.to(stepwise_tgt_logprobs.device)

        repeat_mask = self.ngram_repeat_mask(pred_tokens)
        # blank out timesteps that correspond to @sep@ token
        # also start, end and pad index
        special_tokens = [self.sep_index, self.pad_index, self._end_index, self._start_index]
        # with invert=True, we get tokens OTHER than special tokens as True
        special_token_mask = np.isin(pred_tokens.cpu().numpy(), special_tokens, invert=True)

        repeat_mask = repeat_mask * special_token_mask
        repeat_mask = torch.tensor(repeat_mask, dtype=stepwise_tgt_logprobs.dtype, requires_grad=False)
        repeat_mask = repeat_mask.to(stepwise_tgt_logprobs.device)

        one_minus_probs = torch.clamp((1.0 - stepwise_tgt_logprobs.exp()), min=1e-10)
        # apply repeat mask to blank out only keep the decoding time steps where there is repetitions
        # one_minus_probs = one_minus_probs * repeat_mask

        # [batch_size x 1]
        loss = - (torch.log(one_minus_probs) * repeat_mask * pad_tokens_mask).sum(-1)
        loss = loss.mean() # batch averaging

        return loss

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(self._tensor_based_metric.get_metric(
                    reset=reset))  # type: ignore
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(
                    reset=reset))  # type: ignore
            if self._exact_match_metrics:
                all_metrics.update(
                    self._exact_match_metrics.get_metric(reset=reset))

        return all_metrics

    @staticmethod
    def ngram_repeat_mask(xs, n=2):
        """
        https://github.com/facebookresearch/unlikelihood_training/blob/master/custom/sequence_penalty_loss.py
        """
        batch_size, max_tgt_len = xs.size()
        mask = np.zeros((batch_size, max_tgt_len))
        for i, x in enumerate(xs):
            seen = set()
            xl = x.tolist()
            for j in range(len(x)-n):
                ng = tuple(xl[j:j+n])
                if ng in seen:
                    mask[i, j:j+n] = 1
                seen.add(ng)
        return mask

    def _transform_hidden(self, state, step_i):
        """
        Transform decoder h
        h' = Wh
        where h' is the transformed hidden state to predict token step_i ahead in the future 
        """
        h_transformed = self._state_transform_layers[step_i](state["decoder_hidden"])
        return h_transformed

    def _compute_future_attn_scores(self, h_transformed, state, source_mask):
        """
        Assuming dot-product attention on the transformed hidden state h'
        """
        trimmed_encoder_outputs = state["encoder_outputs"][:, 1:-1]
        attn_scores = self._future_attention(h_transformed, trimmed_encoder_outputs, source_mask)

        return attn_scores

    def _get_future_generation_scores(self, h_transformed):
        """
        Logits over output vocabulary for predict token step_i timesteps ahead
        """
        return self._output_generation_layer(h_transformed)

    def _compute_future_step_ll(self, step_i, state, generation_scores_mask, step_target_tokens, target_to_source, copy_mask):
        """
        For each future token to be predicted, compute 
        1) hidden state transformation
        2) attention distribution based on the above transformed state --> get attention context vector
        3) copy_scores (over source vocab): compute attention scores and re-use that as copy scores (from step 2)
        4) generation_scores (over the vocab) : with [attn, h]->[vocab] (using summation here, can use concat instead)
        5) _get_ll_contrib
        """

        # Step 1 : (batch_size, h_dim)
        h_transformed = self._transform_hidden(state, step_i) 
        # Step 2 and 3 : (batch_size, max_src_len)
        attn_scores = self._compute_future_attn_scores(h_transformed, state, copy_mask) # un-normalized scores
        
        # Step 4:
        normalized_attn_scores = util.masked_softmax(attn_scores, copy_mask)
        normalized_attn_scores = normalized_attn_scores.unsqueeze(1)
        # Attention context vector
        attn_ctx = torch.bmm(normalized_attn_scores, state["encoder_outputs"][:, 1:-1]).squeeze(1)
        h_transformed_combined_with_attn = attn_ctx + h_transformed
        generation_scores = self._get_future_generation_scores(h_transformed_combined_with_attn) # un-normalized scores
        
        # Step 5:
        future_step_log_likelihood, _ = self._get_ll_contrib(generation_scores, 
                                             generation_scores_mask,
                                             attn_scores,
                                             step_target_tokens,
                                             target_to_source,
                                             copy_mask
                                             )

        return future_step_log_likelihood, generation_scores, attn_scores

    def _compute_total_future_losses(self, target_tokens, 
                                    future_step_log_likelihoods, 
                                    future_stepwise_log_probs_over_tgt_vocab, 
                                    future_stepwise_log_probs_over_copy_vocab, 
                                    future_stepwise_target_in_source):
        future_losses = dict()

        # 1) MLE Loss
        future_losses["mle"] = defaultdict()
        # For each future token to be predicted
        for step_i in range(1, self.nsteps_ahead+1):
            future_step_log_likelihoods_step_i = torch.cat(future_step_log_likelihoods[step_i], 1)
            target_mask = util.get_text_field_mask(target_tokens)
            # The 0th timestep is just the START token, which is not included in the likelihoods.
            # The 1st timestep is not needed for 1-step future prediction
            target_mask = target_mask[:, step_i+1:].float()
            future_step_log_likelihoods_step_i = future_step_log_likelihoods_step_i * target_mask
            future_ll_step_i = future_step_log_likelihoods_step_i.sum(dim=-1)
            future_losses["mle"][step_i] = - future_ll_step_i.mean()
            # Total future losses from future token(s) prediction
            # Weighted by coeffient which gets halved every step_i; For example:
            # at step-1 => 0.50
            # at step-2 => 0.25
            future_losses["mle"][step_i] = future_losses["mle"][step_i] * (self.future_loss_coefficient/(step_i+1))


        future_losses["mle"] = sum(future_losses["mle"].values())

        # 2) Target Unlikelihood Loss
        if self.future_tgt_ul:
            future_losses["tgt_ul"] = defaultdict()
            for step_i in range(1, self.nsteps_ahead+1):
                future_stepwise_log_probs_over_tgt_vocab[step_i] = torch.stack(future_stepwise_log_probs_over_tgt_vocab[step_i], dim=1)
                future_losses["tgt_ul"][step_i] = self._compute_token_level_unlikelihood_loss(future_stepwise_log_probs_over_tgt_vocab[step_i], 
                                                                                            target_tokens, step_i)
                future_losses["tgt_ul"][step_i] = future_losses["tgt_ul"][step_i] * self.tgt_token_unlikelihood_loss_coefficient
                future_losses["tgt_ul"][step_i] = future_losses["tgt_ul"][step_i] * (self.future_loss_coefficient/(step_i+1))

            future_losses["tgt_ul"] = sum(future_losses["tgt_ul"].values())

        # 3) Copy Unlikelihood Loss
        if self.future_copy_ul:
            future_losses["copy_ul"] = defaultdict()
            for step_i in range(1, self.nsteps_ahead+1):
                future_stepwise_log_probs_over_copy_vocab[step_i] = torch.stack(future_stepwise_log_probs_over_copy_vocab[step_i], dim=1)
                future_stepwise_target_in_source[step_i] = torch.stack(future_stepwise_target_in_source[step_i], dim=1)
                future_losses["copy_ul"][step_i] = self._compute_copy_token_level_unlikelihood_loss(future_stepwise_log_probs_over_copy_vocab[step_i], 
                                                                                                    future_stepwise_target_in_source[step_i])
                future_losses["copy_ul"][step_i] = future_losses["copy_ul"][step_i] * self.copy_token_unlikelihood_loss_coefficient
                future_losses["copy_ul"][step_i] = future_losses["copy_ul"][step_i] * (self.future_loss_coefficient/(step_i+1))

            future_losses["copy_ul"] = sum(future_losses["copy_ul"].values())

        # Total future token prediction loss
        total_future_loss = sum(future_losses.values())
        
        return total_future_loss
