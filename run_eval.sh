# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# SET PATH
export HOME_PATH=absolute/path/to/keyphrase-generation/folder
export EVALSET="kp20k"
export DATASET="kp20k"
export BATCH_SIZE=256

# Specify models configurations that you would like to evaluate
# Un-comment all configurations below if the corresponding models have been trained
EVAL_MODELS=(
    copy_seq2seq_attn_mle_greedy.tgt_15.0.copy_18.0
    # copy_seq2seq_attn_mle_greedy.tgt_15.0.copy_18.0.futureLoss_1.0.nsteps_2
    # copy_seq2seq_attn_mle_greedy.tgt_15.0.copy_18.0.futureLoss_1.0.nsteps_2.future_mle_tgt_copy
)

for BASE_PATH in "${EVAL_MODELS[@]}"
do  
    BASE_PATH=output/$DATASET/$BASE_PATH
    cd output/
    sh create_model_tar.sh $HOME_PATH/$BASE_PATH $HOME_PATH/$BASE_PATH/best.th
    cd ..

    export MODEL_CHECKPOINT=$BASE_PATH/model.tar.gz
    export JSON_SAVE_PATH=$BASE_PATH/test_predictions.json
    export EVAL_SAVE_DIR=$BASE_PATH/eval_results_$EVALSET

    allennlp predict $MODEL_CHECKPOINT data/sample_testset/$EVALSET\_sorted.jsonl \
                    --include-package keyphrase_generation \
                    --predictor seq2seq_attn --cuda-device 0 --batch-size $BATCH_SIZE \
                    --output-file $JSON_SAVE_PATH

    python keyphrase_generation/utils/convert_json_to_txts.py \
                    --json_file_path $JSON_SAVE_PATH \
                    --save_path $EVAL_SAVE_DIR

    find $EVAL_SAVE_DIR/pred.txt -type f -exec sed -i 's/@sep@//g' {} \;

    python keyphrase_generation/evaluation/evaluate_prediction.py -pred_file_path $EVAL_SAVE_DIR/pred.txt \
                                -src_file_path $EVAL_SAVE_DIR/src.txt \
                                -trg_file_path $EVAL_SAVE_DIR/tgt.txt \
                                -exp $EVALSET -export_filtered_pred -disable_extra_one_word_filter -invalidate_unk \
                                -all_ks 5 M -present_ks 5 M -absent_ks 5 M \
                                -exp_path $EVAL_SAVE_DIR \
                                -filtered_pred_path $EVAL_SAVE_DIR \
                                -meng_rui_precision

done

for BASE_PATH in "${EVAL_MODELS[@]}"
do
    echo $BASE_PATH
    export BASE_PATH=output/$DATASET/$BASE_PATH
    export EVAL_SAVE_DIR=$BASE_PATH/eval_results_$EVALSET

    # Quality Eval Results
    python -u keyphrase_generation/evaluation/parse_results.py --results_file $EVAL_SAVE_DIR/results_log_5_M_5_M_5_M_meng_rui_precision.tsv > $EVAL_SAVE_DIR/quality_eval.txt
    cat $EVAL_SAVE_DIR/quality_eval.txt
    # Diversity Eval Results
    python -u keyphrase_generation/evaluation/diversity_eval.py --pred_file $EVAL_SAVE_DIR/pred.txt > $EVAL_SAVE_DIR/diversity_eval.txt
    cat $EVAL_SAVE_DIR/diversity_eval.txt
done
