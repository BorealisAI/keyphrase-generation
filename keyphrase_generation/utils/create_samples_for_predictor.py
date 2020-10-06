# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
AllenNLP Seq2Seq predictor requires the samples to be in a JSONL format
with each line containing two fields: {"source": seq1, "target": seq2}

Convert JSON output from Seq2Seq predictor to txt format for evaluation 
"""

import os
import json
import argparse
import logging
import random

from typing import Tuple, List, Dict

from tqdm import tqdm
from allennlp.common.checks import ConfigurationError
from keyphrase_generation.utils.common_utils import END_OF_TITLE

logger = logging.getLogger(__name__)


def main(file_path: str,
         save_file_name: str,
         source_field_names: List[str],
         target_field_name: str,
         delimiter: str,
         count: int,
         seed: int):

    with open(file_path, 'r') as fin:
        logger.info("Reading instances from lines in file at: %s", file_path)

        source_seqs = []
        target_seqs = []

        for line in tqdm(enumerate(fin)):
            
            source_sequence, target_sequence = read_keyphrase_jsonl_line(line,
                                                                        source_field_names,
                                                                        target_field_name,
                                                                        delimiter)

            source_seqs.append(source_sequence)
            target_seqs.append(target_sequence)

    if count != -1:
        logger.info("Sampling %d from %d records", count, len(source_seqs))
        random.seed(seed)
        indices = random.sample(range(len(source_seqs)), count)
    else:
        indices = list(range(len(source_seqs)))

    save_path = os.path.join("data/sample_testset", save_file_name)
    with open(save_path, 'w') as fout:
    
        logger.info("Saving predictor sample file as: %s", save_path)
    
        for idx in tqdm(indices):
            line = {
                "source": source_seqs[idx],
                "target": target_seqs[idx]
            }
            fout.write(json.dumps(line))
            fout.write("\n")


def read_keyphrase_jsonl_line(line: Tuple[int, Dict],
                              source_field_names: List[str],
                              target_field_name: str,
                              delimiter: str,
                              keyphrase_sep_symbol: str = ';'):
    line_num = line[0]
    row = json.loads(line[1])

    all_fields = source_field_names + [target_field_name]
    if not set(all_fields).issubset(row.keys()):
        raise ConfigurationError(
            "Invalid line format: %s (line number %d) - "
            "Incorrect field names specified for source/target in the json file" % (
                row, line_num + 1)
        )

    joiner = ' ' + END_OF_TITLE + ' '
    source_sequence = joiner.join([row.get(key) for key in source_field_names])
    
    target_sequence = row[target_field_name]
    if isinstance(target_sequence, str) and delimiter:
        # if keywords are provided as a string delimited by ';'
        target_sequence = target_sequence.split(delimiter)
    # if keywords are provided as a list
    joiner = ' ' + keyphrase_sep_symbol + ' '
    target_sequence = joiner.join(target_sequence)
    
    return source_sequence.lower(), target_sequence.lower()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', '-fp', type=str, required=True,
                        help="File path of the original test set")
    parser.add_argument('--save_file_name', '-sfn', type=str, required=True,
                        help="File name to save the created JSONL - in the directory data/sample_testset/")
    parser.add_argument('--source_field_names', '-s', type=str, nargs='+', required=True,
                        help="The name of the attribute that corresponds to the input sequence")
    parser.add_argument('--target_field_name', '-t', type=str, required=True,
                        help="The name of the attribute that corresponds to the output, i.e., the keywords")
    parser.add_argument('--delimiter', '-d', type=str, default=';',
                        help="The delimiter used to separate the set of keywords in the target field")
    parser.add_argument('--count', '-n', type=int, default=100,
                        help="Number of instances to be sampled from the original test set \
                            Set to -1 if no sampling is to be used.")
    parser.add_argument('--seed', '-seed', type=int, default=10,
                        help="Random seed for sampling")

    args = parser.parse_args()
    main(**vars(args))
