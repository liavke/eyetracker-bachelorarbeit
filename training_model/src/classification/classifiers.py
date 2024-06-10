from config import Classifier
from sklearn.svm import SVC # "Support vector classifier"
import numpy as np


class SVM_Classifier(Classifier):
    def __init__(self) -> None:
        super().__init__()
        self.model = SVC(kernel='linear', C=1E10) #or other

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self):
        pass

    def _set_feature_n(self, n):
        return self.X.iloc[:n]


class Decision_Trees_Classifier(Classifier):
    def __init__(self) -> None:
        super().__init__()

    def fit(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

    def _set_feature_n(self):
        pass

class K_Means_Classifier(Classifier):
    def __init__(self) -> None:
        super().__init__()

    def fit(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

    def _set_feature_n(self):
        pass

class Naive_Bayes_Classifier(Classifier):
    def __init__(self) -> None:
        super().__init__()

    def fit(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

    def _set_feature_n(self):
        pass
