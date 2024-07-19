import sys
import os
sys.path.append(os.getenv('PATH_TO_CLASSIFICATION'))

from src.classification.config import BaseClassifier, ClassifiersConfig

import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC # "Support vector BaseClassifier"
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

import src.classification.utils as utils

class MultiBaseClassifiers(BaseClassifier):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        self.train_test_data = train_test_split(self.X, self.y, test_size=0.33, random_state=42, shuffle=True)

        self.svm_model = utils.find_best_svm(x_train=self.train_test_data[0], 
                                               y_train=self.train_test_data[1],
                                               config = ClassifiersConfig.svc_config)
        
        self.trees_model = utils.find_best_tree(x_train=self.train_test_data[0], 
                                                 y_train=self.train_test_data[1],
                                                 config = ClassifiersConfig.dt_config)
        
        self.rf_model = utils.find_best_forest(x_train=self.train_test_data[0], 
                                              y_train=self.train_test_data[1],
                                              config = ClassifiersConfig.rf_config)
        self.nb_model = GaussianNB()
    
    def run(self, binary=False):
        X_train, X_test, y_train, y_test = self.train_test_data
        self.fit(X_train=X_train, y_train=y_train)        
        return self.evaluate(X_test=X_test, y_test=y_test)

    def fit(self, X_train, y_train):
        # defining parameter range 
        print("### TRAINING SVM ###")
        self.svm_model.fit(X_train, y_train)
        print("### TRAINING TREES ###")
        self.trees_model.fit(X_train, y_train)
        print("### TRAINING NAIV BAYES ###")
        self.nb_model.fit(X_train,y_train)


    def predict(self, X_test):
        return {
            "SVM" : self.svm_model.predict(X=X_test),
            "Trees": self.trees_model.predict(X_test),
            "NaiveBayes": self.nb_model.predict(X_test),
        }
    """
    def predict_prob(self,X_test):
        return {
            "SVM" : self.svm_model.predict_proba(X=X_test),
            "Trees": self.trees_model.predict_proba(X_test),
            "NaiveBayes": self.nb_model.predict_proba(X_test),
        }
        """

    def evaluate(self, X_test, y_test):
        evaluations = pd.DataFrame()
        raw_results = pd.DataFrame()
        roc_auc_list = []

        predictions = self.predict(X_test)
        #prediction_probablities = self.predict_prob(X_test)
        

        for model_name, pred in predictions.items():
            num_y , num_pred = self._turn_labels_numeric(y_test, pred)
            score_df = pd.DataFrame({
                "model_name" : model_name,
                "eer_score" : [utils.calculate_eer(y_test=y_test, predictions=pred)],
                "f1_score" : [f1_score(y_pred=pred, y_true=y_test, average='micro')],
                #"rocauc_score" :roc_auc_score(y_score=num_pred, y_true=num_y, multi_class='ovr')
                })
            evaluations = pd.concat([evaluations, score_df])

            raw_results[model_name] = pred
        """
        for item in prediction_probablities.values():
            rocauc_score = roc_auc_score(y_score=item, y_true=y_test, multi_class='ovr')
            roc_auc_list.append(rocauc_score)

        evaluations["rocauc_score"] = roc_auc_list"""

        raw_results['ground truth'] = y_test
        return evaluations, raw_results

    def _set_svm(self):
        param_grid = [
            {'C': [0.1, 1, 10, 100, 1000], 
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
            'kernel': ['rbf']},
            {'C': [0.1, 1, 10, 100, 1000], 
            'kernel': ['poly'], 
            'degree': [2, 3, 4]},
            {'C': [0.1, 1, 10, 100, 1000], 
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
            'kernel': ['sigmoid'], 
            'coef0': [0, 0.1, 0.5, 1]},
            {'C': [0.1, 1, 10, 100, 1000], 
            'kernel': ['linear']}
        ]
        return GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy',refit = True, verbose = 3)
    
    def find_best_params(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42, shuffle=True)
        param_grid = [
            {'C': [0.1, 1, 10, 100, 1000], 
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
            'kernel': ['rbf']},
            {'C': [0.1, 1, 10, 100, 1000], 
            'kernel': ['poly'], 
            'degree': [2, 3, 4]},
            {'C': [0.1, 1, 10, 100, 1000], 
            'kernel': ['sigmoid'], 
            'coef0': [0, 0.1, 0.5, 1]},
            {'C': [0.1, 1, 10, 100, 1000], 
            'kernel': ['linear']}
        ]
        grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy',refit = True, verbose = 3)
        grid.fit(X_train, y_train)
        return grid.best_params_
    
    def _turn_labels_numeric(self, y_test, pred):
        numeric_labels_dict = {'self':1, 'deepfake':2, 'other':3}
        y_new = [numeric_labels_dict[label] for label in y_test]
        pred_new = [numeric_labels_dict[label] for label in pred]
        return y_new, pred_new
    
    def find_best_svm(self,x_train, y_train, config):
        grid = GridSearchCV(estimator=SVC(), param_grid=config, cv=5, scoring='accuracy',refit = True, verbose = 3)
        grid.fit(x_train, y_train)
        params = grid.best_params_
        SVC(co)

    def find_best_tree(self,x_train, y_train, config):
        grid = GridSearchCV(estimator=SVC(), param_grid=config, cv=5, scoring='accuracy',refit = True, verbose = 3)
        grid.fit(x_train, y_train)
        params = grid.best_params_
        SVC(co)

    def find_best_forest(self,x_train, y_train, config):
        grid = GridSearchCV(estimator=SVC(), param_grid=config, cv=5, scoring='accuracy',refit = True, verbose = 3)
        grid.fit(x_train, y_train)
        params = grid.best_params_
        SVC(co)

    def _set_feature_n(self):
        pass

    def _get_best_feature(self):
        pass

class BinaryBaseClassifiers(BaseClassifier):
    def __init__(self, X, y) -> None:
        self.svm_model = SVC(kernel='linear', C=1E10, probability=True)
        self.trees_model = tree.DecisionTreeClassifier()
        #self.kmeans_model = KMeans(n_clusters=3)
        self.nb_model = GaussianNB()
        self.X = X
        self.y = y
    
    def run(self, binary=False):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42, shuffle=True)
        self.fit(X_train=X_train, y_train=y_train)        
        return self.evaluate(X_test=X_test, y_test=y_test)


    def fit(self, X_train, y_train):
        print("### TRAINING SVM ###")
        self.svm_model.fit(X_train, y_train)
        print("### TRAINING TREES ###")
        self.trees_model.fit(X_train, y_train)
        print("### TRAINING NAIV BAYES ###")
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
        raw_results = pd.DataFrame()
        roc_auc_list = []

        predictions = self.predict(X_test)
        prediction_probablities = self.predict_prob(X_test)

        for model_name, pred in predictions.items():
            eer,_, _, _ = utils.eer(ground_truth=y_test, predictions=pred)
            score_df = pd.DataFrame({
                "model_name" : model_name,
                "eer_score" : [eer],
                "f1_score" : [f1_score(y_pred=pred, y_true=y_test, average='micro')]
                })
            evaluations = pd.concat([evaluations, score_df])

            raw_results[model_name] = pred

        for item in prediction_probablities.values():
            rocauc_score = roc_auc_score(y_score=item, y_true=y_test, multi_class='ovr')
            roc_auc_list.append(rocauc_score)

        evaluations["rocauc_score"] = roc_auc_list
        raw_results['ground truth'] = y_test
        return evaluations, raw_results