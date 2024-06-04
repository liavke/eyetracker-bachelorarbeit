from config import Classifier
from sklearn.svm import SVC # "Support vector classifier"


class SVM_Classifier(Classifier):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y
        self.model = SVC(kernel='linear', C=1E10) #or other

    def fit(self):
        self.model.fit(self.X, self.y)

    def predict(self):
        pass

class Decision_Trees_Classifier(Classifier):
    def __init__(self) -> None:
        super().__init__()

    def fit(self):
        pass

    def predict(self):
        pass

class K_Means_Classifier(Classifier):
    def __init__(self) -> None:
        super().__init__()

    def fit(self):
        pass

    def predict(self):
        pass

class Naive_Bayes_Classifier(Classifier):
    def __init__(self) -> None:
        super().__init__()

    def fit(self):
        pass

    def predict(self):
        pass