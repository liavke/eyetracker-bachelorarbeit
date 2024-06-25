import sys
import os
sys.path.append(os.getenv('PATH_TO_CLASSIFICATION'))

class BaseClassifier():
    def __init__(self, X, y) -> None:
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

    def _set_feature_n(self):
        pass

    def _get_best_feature(self):
        pass