# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert single output .json from Seq2SeqPredictor to .txt files for:
- src.txt
- tgt.txt
- pred.txt
"""
import os
import json
import argparse
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


def main(json_file_path: str, save_path) -> None:

    with open(json_file_path, 'r') as fin:
        logger.info(
            "Reading instances from lines in JSONL file at: %s", json_file_path)

        source_seqs = []
        target_seqs = []
        pred_seqs = []

        for line in tqdm(enumerate(fin)):
            line_num = line[0]
            row = json.loads(line[1])

            all_fields = ["source", "target", "pred"]
            if not set(all_fields).issubset(row.keys()):
                raise ConfigurationError(
                    "Invalid line format: %s (line number %d) - "
                    "Incorrect field names specified for source/target in the json file" % (
                        row, line_num + 1)
                )

            source_seqs.append(row["source"])
            target_seqs.append(row["target"])
            pred_seqs.append(row["pred"])

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger.info("Saving as text files ...") 
    save_as_txt(source_seqs, os.path.join(save_path, "src.txt"))
    save_as_txt(target_seqs, os.path.join(save_path, "tgt.txt"))
    save_as_txt(pred_seqs, os.path.join(save_path, "pred.txt"))


def save_as_txt(sequences, save_file_name):
    with open(save_file_name, 'w') as fout:
        for seq in tqdm(sequences):
            fout.write(seq + "\n")
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_path', type=str, required=True,
                        help="Path to the json file output by the Seq2SeqPredictor")
    parser.add_argument('--save_path', type=str, required=True,
                        help="Folder to save the txt output files")

    args = parser.parse_args()
    main(**vars(args))
