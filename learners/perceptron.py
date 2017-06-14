"""
    Identical to Perceptron implementation of Chen 2012.
"""

import numpy as np


def _snorm(w):
    return np.linalg.norm(w)


class Perceptron(object):

    def __init__(self, classes):
        if set(classes) != set([-1.0, 1.0]):
            raise ValueError
        self.w = None
        self.score = 0

    def reset(self, x):
        self.w = np.zeros(x.shape)

    def raw_predict(self, x):
        if self.w is None:
            self.reset(x)

        prod = self.w * x.T
        norm = _snorm(self.w)
        if norm > 1.0:
            return prod / norm
        return prod

    def predict(self, x):
        self.score = self.raw_predict(x)
        if self.score > 0.0:
            return 1.0
        return -1.0

    def partial_fit(self, x, y, sample_weight=1.0):
        if self.raw_predict(x) * y <= 0.0:
            self.w = self.w + sample_weight * y * x

    def get_score(self):
        return self.score
