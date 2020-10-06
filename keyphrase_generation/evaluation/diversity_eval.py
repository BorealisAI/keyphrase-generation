# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import numpy as np

from tqdm import tqdm
from nltk.stem.porter import *

stemmer = PorterStemmer()

import nltk
import sent2vec

from typing import List
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from rapidfuzz import fuzz
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction

from keyphrase_generation.evaluation.string_helper import stem_word_list,stem_str_2d_list


def read_from_txt(filepath):
    """
    Read targets or model outputs from text files
    """
    lines = []
    with open(filepath, 'r') as fio:
        for line in tqdm(fio):
            lines.append(line.strip())
            
    return lines

def kps_to_tokens(all_kps):
    """
    Convert list of keyphrase sequences to list of word lists
    """
    kp_token_list = []
    for item in all_kps:
        kps = item.split(';')
        words = []
        for kp in kps:
            words.append(kp.split(' '))
        kp_token_list.append(words)

    return kp_token_list

def read_and_process(filepath):
    """
    Given a txt file path, read lines, convert tokens and stem the tokens
    """
    keyphrases = read_from_txt(filepath)
    keyphrases = stem_str_2d_list(kps_to_tokens(keyphrases))

    return keyphrases

def compute_stats(keyphrases):
    """
    For each row/records:
    - Count of number of keyphrases 
    - Ratio of duplicate keyphrases
    - Ratio of duplicate unigrams
    """

    counts = []
    kp_duplication = []
    token_duplication = []
    
    for item_list in tqdm(keyphrases):
        # number of keyphrases
        counts.append(len(item_list))
        
        # duplicate keyphrases
        kp_string = [' '.join(item) for item in item_list]
        duplication_ratio = 1.0 - len(set(kp_string))/len(kp_string)
        kp_duplication.append(duplication_ratio)
        
        # duplicate unigrams
        tokens = [word for kp in item_list for word in kp]
        duplication_ratio = 1.0 - len(set(tokens))/len(tokens)
        token_duplication.append(duplication_ratio)
    
    return counts, kp_duplication, token_duplication

def calc_bleu(references, hypothesis, weight):
    return nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weight,
                                                   smoothing_function=SmoothingFunction().method1)

def selfbleu_scorer(kps: List[str], ngram=3):
    """
    Arguments:
        kps: list of keyphrases
    
    Returns:
        avg_self_bleu: average pairwaise Self BLEU between keyphrases 
    """
    
    weights = tuple((1. / ngram for _ in range(ngram)))
    bleu_list = []
    
    n_kps = len(kps)
    
    for i in range(n_kps):
        # use ith element as hypothesis and the rest as references
        hypothesis = word_tokenize(kps[i])        
        references = kps[:i] + kps[i+1:] # skip the ith element
        references = [word_tokenize(r) for r in references]
        
        bleu_list.append(calc_bleu(references, hypothesis, weights))

    avg_self_bleu = np.mean(bleu_list)
    
    return avg_self_bleu

def edit_distance_scorer(kps: List[str]):
    """
    Arguments:
        kps: list of keyphrases
    
    Returns:
        avg_edit_dist: average pairwaise edit distance between keyphrases 
    """
    
    edit_dists = []
    n_kps = len(kps)
    
    for i in range(n_kps):
        # use ith element as source and the rest as targets
        source = kps[i]
        targets = kps[:i] + kps[i+1:] # skip the ith element        
        target_dists = [] # to store distance of source to each of the targets
        for t in targets:
            target_dists.append(fuzz.ratio(source, t))
        edit_dists.append(np.mean(target_dists))

    avg_edit_dist = np.mean(edit_dists)
    return avg_edit_dist

def embedding_similarity_scorer(kps: List[str], model):
    """
    Arguments:
        kps: list of keyphrases
    
    Returns:
        avg_emb_sim: average pairwaise embedding similarity between keyphrases 
    """
    
    n_kps = len(kps)
    embs = model.embed_sentences(kps)
    embs_sparse = sparse.csr_matrix(embs)
    similarities = cosine_similarity(embs_sparse)
    avg_emb_sim = np.mean(similarities[np.triu_indices(n_kps, k = 1)])

    return avg_emb_sim

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="Path to pred.txt file, each line has kps separated by ;")
    parser.add_argument("--sent2vec_path", type=str, default='data/sent2vec/wiki_unigrams.bin', help="Path to sent2vec embeddings /path/to/wiki_unigrams.bin ")

    args = parser.parse_args()

    model = sent2vec.Sent2vecModel()
    model.load_model(args.sent2vec_path)
    
    # Read as list of lists
    all_keyphrases = read_and_process(args.pred_file)
    counts, kp_duplication, token_duplication = compute_stats(all_keyphrases)
        
    metrics = [np.mean(counts), np.mean(kp_duplication)*100, np.mean(token_duplication)*100]
    print(' Count of number of keyphrases  \t= {:.2f} \n Avg.Ratio of duplicate keyphrases \t= {:.2f} \n Avg.Ratio of duplicate unigrams \t= {:.2f}'.format(
        *metrics))
    print('-'*50)

    # Initialize metric lists
    selfbleu_scores = []
    edit_dist_scores = []
    emb_sim_scores = []
    src_pred_sim_scores = []
    
    # Read as string
    all_keyphrases = read_from_txt(args.pred_file)
    
    for i, item in tqdm(enumerate(all_keyphrases)):
        kps = item.split(';')

        if len(kps) > 1:
            sb = selfbleu_scorer(kps)
            selfbleu_scores.append(sb)

            ed = edit_distance_scorer(kps)
            edit_dist_scores.append(ed)

            es = embedding_similarity_scorer(kps, model)
            emb_sim_scores.append(es)

    metrics = [np.mean(selfbleu_scores) * 100, np.mean(edit_dist_scores), np.mean(emb_sim_scores)]
    print(' Self BLEU  \t= {:.2f} \n Edit Distance \t= {:.1f} \n Inter KP Embedding Similarity \t= {:.3f}'.format(*metrics))
    print('-'*50)
