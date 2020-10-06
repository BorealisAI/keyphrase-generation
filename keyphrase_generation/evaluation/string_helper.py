# MIT License

# Copyright (c) 2020 Hou Pong Chan

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
####################################################################################
# Code is taken from the catSeqTG-2RF1 model (https://arxiv.org/abs/1906.04106) implementation
# from https://github.com/kenchan0226/keyphrase-generation-rl by Hou Pong Chan
####################################################################################

from nltk.stem.porter import *
stemmer = PorterStemmer()


def stem_str_list(str_list):
    # stem every word in a list of word list
    # str_list is a list of word list
    stemmed_str_list = []
    for word_list in str_list:
        stemmed_word_list = stem_word_list(word_list)
        stemmed_str_list.append(stemmed_word_list)
    return stemmed_str_list

def stem_str_2d_list(str_2dlist):
    """
    stem every word in a list of word list
    """
    stemmed_str_2dlist = []
    for str_list in str_2dlist:
        stemmed_str_list = [stem_word_list(word_list) for word_list in str_list]
        stemmed_str_2dlist.append(stemmed_str_list)
    return stemmed_str_2dlist

def stem_word_list(word_list):
    """
    Stems words in the provided list
    """
    return [stemmer.stem(w.strip().lower()) for w in word_list]
