// Copyright (c) 2020-present, Royal Bank of Canada.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

{
    "dataset_reader": {
        "type": "keyphrase_seq2seq_copynet_jsonl_reader",
        "json_source_field_names": ["title", "abstract"],
        "json_target_field_name": "keyword",
        "json_target_field_delimiter": ";",
        "target_namespace": "target_tokens", 
        "lazy": true,
        "source_max_tokens": 300,
        "target_max_tokens": 30,
        "source_tokenizer": {
            "type": "word",
            "word_splitter": {
                "type": "spacy"
            },
            "word_filter": {
                "type": "digit_filter",
                "digit_pattern": ".*\\d.*",
                "word_pattern": "(\\w|\\.|\\,|\\@)"  // @ is kept because it is used in the <EOT> token in the source seq
            }
        },
        
        // if STOPWORDS and PUNCTUATIONS need to be removed, then they need to be added to a text file
        // and link it via the "source_tokenizer" --> "word_filter" in the above config
        // can also be a REGEX filter 
        // based on allennlp.data.tokenizers.word_filter.StopwordFilter
        // Creating a new filter class instead
        "target_tokenizer": {
            "type": "word",
            "word_splitter": {
                "type": "spacy"
            },
            "word_filter": {
                "type": "digit_filter",
                "digit_pattern": ".*\\d.*",
                "word_pattern": "(\\w|\\.|\\,|\\@)"  // @ is kept because it is used in the <SEP> token in the target seq
            }
        },
        "source_token_indexers": {
            "tokens": {
                "type": "single_id",
                "namespace": "source_tokens",
                "lowercase_tokens": true
            }
        },
    },

    "vocabulary": {
        "max_vocab_size": { 
            // if source and target vocabs are not shared, specify two different namespaces here
            "source_tokens": 50000,
            "target_tokens": 10000
        },
        "tokens_to_add": {
            "target_tokens": ["@COPY@"]
        }
    },

    "train_data_path": "data/KPTimes/kptimes_sorted/train.jsonl",
    "validation_data_path": "data/KPTimes/kptimes_sorted/valid500.jsonl",
    "test_data_path": "data/KPTimes/kptimes_sorted/test500.jsonl",

    "model": {
        "type": "copynet_seq2seq_attn",
        "max_decoding_steps": 30,
        "beam_size": 1,
        "source_namespace": "source_tokens",
        "target_namespace": "target_tokens",
        "sampling_strategy": "greedy", 
        "prev_context_len": 0,
        "seq_ul_coefficient": 0.0,
        "start_fine_tune_iter": 1000000,
        "tgt_token_unlikelihood_loss_coefficient": 15.0, # unlikelihood training rank_alpha (set to 1.0 in the paper codebase)
        "copy_token_unlikelihood_loss_coefficient": 18.0, 
        "nsteps_ahead": 2,
        "future_loss_coefficient": 1.0,
        "future_tgt_ul": false,
        "future_copy_ul": false,
        "entropy_reg_coefficient": 0.0,
        "scheduled_sampling_ratio": 0.0,
        "lower_ss_ratio_every": 1000000,
        "target_embedding_dim": 100,
        "source_embedder": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 100,
                "trainable": true,
                "max_norm": 1.0,
                "vocab_namespace": "source_tokens"
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 100,
            "hidden_size": 150,
            "bidirectional": true,
            "num_layers": 1,
            "dropout": 0.2
        },
        // for decoder, allenNLP seq2seq currently only has LSTM cell for RNN decoder
        "attention": {
            "type": "dot_product"
            },
        },

    "iterator": {
        "type": "bucket",
        "batch_size": 64,
        "max_instances_in_memory": 10000,
        "sorting_keys": [["source_tokens", "num_tokens"]]
    },

    "trainer": {
        "cuda_device": -1,
        "num_epochs": 10,
        "learning_rate_scheduler": {
          "type": "reduce_on_plateau",
          "factor": 0.5,
          "mode": "max",
          "patience": 10
        },
        "optimizer": {
          "lr": 0.001,
          "type": "adam",
          "weight_decay": 0.0001
        },
        "patience": 10,
        "validation_metric": "+F-score",
        "grad_clipping": 1.0
    }
}