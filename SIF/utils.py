from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

from scipy.stats import pearsonr
import numpy as np


class WordToWeight(object):
    def __init__(self, path):
        self._vocab = {}
        with open(path, 'r') as f:
            for line in f:
                index = line.find(' ')
                self._vocab[line[:index]] = line[index + 1:]

    def __len__(self):
        return len(self._vocab)

    def __getitem__(self, key):
        if isinstance(self._vocab[key], str):
            self._vocab[key] = np.array([float(x) for x in self._vocab[key].split(' ')])
        return self._vocab[key]

    def __contains__(self, key):
        return key in self._vocab

    def keys(self):
        return self._vocab.keys()


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)


def load_sts(file_path):
    sentences1 = []
    sentences2 = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            score, s1, s2 = line.split('\t')
            if not score:
                continue
            labels.append(float(score))
            sentences1.append(s1)
            sentences2.append(s2)
    return sentences1, sentences2, labels


def evaluate(file_path, embed_func, embed_dict):
    s1, s2, labels = load_sts(file_path)
    M1 = embed_func(s1, embed_dict)
    M2 = embed_func(s2, embed_dict)
    pred = [cosine_similarity(vs0, vs1) for vs0, vs1 in zip(M1, M2)]
    return pearsonr(pred, labels)[0]
