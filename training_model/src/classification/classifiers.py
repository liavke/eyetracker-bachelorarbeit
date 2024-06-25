import sys
import os
sys.path.append(os.getenv('PATH_TO_CLASSIFICATION'))

from src.classification.config import BaseClassifier

import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.svm import SVC # "Support vector BaseClassifier"
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import src.classification.utils as utils

class MultiBaseClassifiers(BaseClassifier):
    def __init__(self, X, y) -> None:
        self.svm_model = SVC(kernel='linear', C=1E10, probability=True)
        self.trees_model = tree.DecisionTreeClassifier()
        #self.kmeans_model = KMeans(n_clusters=3)
        self.nb_model = GaussianNB()
        self.X = X
        self.y = y
    
    def run(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42, shuffle=True)
        self.fit(X_train=X_train, y_train=y_train)
        return self.evaluate(X_test=X_test, y_test=y_test)


    def fit(self, X_train, y_train):
        self.svm_model.fit(X_train, y_train)
        self.trees_model.fit(X_train, y_train)
        #self.kmeans_model.fit(X_train)
        self.nb_model.fit(X_train,y_train)


    def predict(self, X_test):
        return {
            "SVM" : self.svm_model.predict(X=X_test),
            "Trees": self.trees_model.predict(X_test),
            "NaiveBayes": self.nb_model.predict(X_test),
        }
    
    def predict_prob(self,X_test):
        return {
            "SVM" : self.svm_model.predict_proba(X=X_test),
            "Trees": self.trees_model.predict_proba(X_test),
            "NaiveBayes": self.nb_model.predict_proba(X_test),
        }

    def evaluate(self, X_test, y_test):
        evaluations = pd.DataFrame()
        roc_auc_list = []

        predictions = self.predict(X_test)
        prediction_probablities = self.predict_prob(X_test)

        for model_name, pred in predictions.items():
            score_df = pd.DataFrame({
                "model_name" : model_name,
                "eer_score" : [utils.calculate_eer(y_test=y_test, predictions=pred)],
                "f1_score" : [f1_score(y_pred=pred, y_true=y_test, average='micro')]
                })
            evaluations = pd.concat([evaluations, score_df])

        for item in prediction_probablities.values():
            rocauc_score = roc_auc_score(y_score=item, y_true=y_test, multi_class='ovr')
            roc_auc_list.append(rocauc_score)

        evaluations["rocauc_score"] = roc_auc_list
        return evaluations


    def _set_feature_n(self):
        pass

    def _get_best_feature(self):
        pass


class SVM_BaseClassifier(BaseClassifier):
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


class Decision_Trees_BaseClassifier(BaseClassifier):
    def __init__(self) -> None:
        super().__init__()
        self.model = tree.DecisionTreeBaseClassifier()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, y_test):
        return self.model.predict(y_test)

    def evaluate(self, X_test , y_test):
        prediction = self.predict(X_test)


    def _set_feature_n(self):
        pass

class K_Means_BaseClassifier(BaseClassifier):
    def __init__(self) -> None:
        super().__init__()
        self.model = KMeans(n_clusters=3)

    def fit(self, X_train):
        self.model.fit(X_train)

    def predict(self):
        pass

    def evaluate(self):
        pass

    def _set_feature_n(self):
        pass
    

class Naive_Bayes_BaseClassifier(BaseClassifier):
    def __init__(self) -> None:
        super().__init__()
        self.model = GaussianNB()

    def fit(self,  X_train, y_train):
        self.model.fit( X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self):
        pass

    def _set_feature_n(self):
        pass
