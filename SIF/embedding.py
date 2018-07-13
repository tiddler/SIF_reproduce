from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

from sklearn.decomposition import TruncatedSVD
from .utils import WordToWeight

import numpy as np
import spacy

FREQ_PATH = '../resources/enwiki_vocab_min200.txt'

nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
freq = WordToWeight(FREQ_PATH)
# count the total word in freq
total_word = 0
for token in freq.keys():
    total_word += freq[token]
total_word = total_word[0]


def average_embedding(line, embed_dict, dim=300):
    word_list = [token.lemma_.lower() for token in nlp(line)
                 if not token.is_stop and not token.is_punct and not token.is_space and token.lemma_.lower() in embed_dict]
    doc_length = len(word_list)
    vs = np.zeros(dim)
    if not doc_length:
        return vs
    for token in word_list:
        vs += embed_dict[token]
    return vs / doc_length


def AVG_embedding(lines, embed_dict, dim=300):
    return np.array([average_embedding(line, embed_dict, dim) for line in lines])


def weighted_embedding(line, embed_dict, dim=300, a=1e-3):
    word_list = [token.lemma_.lower() for token in nlp(line)
                 if not token.is_stop and not token.is_punct and not token.is_space and token.lemma_.lower() in embed_dict]
    doc_length = len(word_list)
    vs = np.zeros(dim)
    if not doc_length:
        return vs
    for token in word_list:
        token_freq = freq[token][0] if token in freq else 1
        a_value = a / (a + token_freq / total_word)
        vs += embed_dict[token] * a_value
    return vs / doc_length


def W_embedding(lines, embed_dict, dim=300, a=1e-3):
    matrix = np.array([weighted_embedding(line, embed_dict, dim, a)
                       for line in lines])
    return matrix


def WR_embedding(lines, embed_dict, dim=300, a=1e-3):
    matrix = np.array([weighted_embedding(line, embed_dict, dim, a)
                       for line in lines])
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(matrix)
    pc = svd.components_
    return matrix - matrix @ pc.T @ pc
