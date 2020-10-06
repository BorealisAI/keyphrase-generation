# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
To parse the evaluation output file and print results on screen
"""

import argparse
import pandas as pd

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True, help="Path to tsv results file")

    args = parser.parse_args()
    df_results = pd.read_table(args.results_file, sep="\t")

    cols_to_display = [
        "macro_avg_p@M_all", "macro_avg_r@M_all", 
        "macro_avg_f1@M_all"
    ]
    print("-"*50)
    for col in cols_to_display:
        print("{}\t{:.3f}".format(col, df_results.loc[0, col]))
    print("-"*50)