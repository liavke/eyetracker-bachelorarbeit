import sys
import os
sys.path.append(os.getenv('PATH_TO_CLASSIFICATION'))

from src.classification.config import BaseClassifier, ClassifiersConfig
from datetime import datetime
import logging
logging.basicConfig(filename=f'src/logs/{datetime.now()}_best_params.log', level=logging.INFO, format='%(asctime)s - %(message)s')

import pandas as pd

import plotly.express as px
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC # "Support vector BaseClassifier"
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

import src.classification.utils as utils

class MultiBaseClassifiers(BaseClassifier):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        self.train_test_data = train_test_split(self.X, self.y, test_size=0.33, random_state=42, shuffle=True)

        self.svm_model = SVC(C=1000, kernel='rbf', gamma=1, probability=True)#self._find_best_svm(x_train=self.train_test_data[0], y_train=self.train_test_data[2],  config = ClassifiersConfig.svc_config)
        
        self.trees_model = tree.DecisionTreeClassifier(max_depth=100, min_samples_leaf=5, min_samples_split=10)#self._find_best_tree(x_train=self.train_test_data[0], y_train=self.train_test_data[2], config = ClassifiersConfig.dt_config)
        
        self.rf_model = RandomForestClassifier(max_depth=80, min_samples_leaf=5, min_samples_split=10, n_estimators=100)#self._find_best_forest(x_train=self.train_test_data[0], y_train=self.train_test_data[2], config = ClassifiersConfig.rf_config)
        self.nb_model = GaussianNB()
    
    def run(self, binary=False):
        X_train, X_test, y_train, y_test = self.train_test_data
        self.fit(X_train=X_train, y_train=y_train)        
        return self.evaluate(X_test=X_test, y_test=y_test, y_train=y_train)

    def fit(self, X_train, y_train):
        # defining parameter range 
        print("### TRAINING SVM ###")
        self.svm_model.fit(X_train, y_train)
        print("### TRAINING TREES ###")
        self.trees_model.fit(X_train, y_train)
        print("### TRAINING FORESTS ###")
        self.rf_model.fit(X_train, y_train)
        print("### TRAINING NAIV BAYES ###")
        self.nb_model.fit(X_train,y_train)


    def predict(self, X_test):
        return {
            "SVM" : self.svm_model.predict(X=X_test),
            "Trees": self.trees_model.predict(X_test),
            "Forests": self.rf_model.predict(X_test),
            "NaiveBayes": self.nb_model.predict(X_test),
        }
    
    def predict_prob(self,X_test):
        return {
            "SVM" : self.svm_model.predict_proba(X=X_test),
            "Trees": self.trees_model.predict_proba(X_test),
            "Forests": self.rf_model.predict_proba(X_test),
            "NaiveBayes": self.nb_model.predict_proba(X_test),
        }
        

    def evaluate(self, X_test, y_test, y_train):
        evaluations = pd.DataFrame()
        raw_results = pd.DataFrame()
        roc_auc_list = []

        predictions = self.predict(X_test)
        prediction_probablities = self.predict_prob(X_test)
        
        for model_name, pred in predictions.items():
            num_y , num_pred = self._turn_labels_numeric(y_test, pred)
            #percision, recall = utils.calculate_percision_recall(y_test=y_test, predictions=pred)
            score_df = pd.DataFrame({
                "model_name" : model_name,
                "accuracy": [accuracy_score(y_true=y_test, y_pred=pred)],
                "eer_score" : [utils.calculate_eer(y_test=y_test, predictions=pred)],
                "f1_score" : [f1_score(y_pred=pred, y_true=y_test, average='macro')],
                #"percision":[percision],
                #"recall":[recall]
                })
            evaluations = pd.concat([evaluations, score_df])

            raw_results[model_name] = pred
        
        for item in prediction_probablities.values():
            rocauc_score = roc_auc_score(y_score=item, y_true=y_test, multi_class='ovr', average='macro')
            roc_auc_list.append(rocauc_score)

        evaluations["rocauc_score"] = roc_auc_list

        raw_results['ground truth'] = y_test
        return evaluations, raw_results
    
    def _turn_labels_numeric(self, y_test, pred):
        numeric_labels_dict = {'self':1, 'deepfake':2, 'other':3}
        y_new = [numeric_labels_dict[label] for label in y_test]
        pred_new = [numeric_labels_dict[label] for label in pred]
        return y_new, pred_new
    
    def _find_best_svm(self,x_train, y_train, config):
        grid = GridSearchCV(estimator=SVC(), param_grid=config, cv=5, scoring='accuracy',refit = True, verbose = 3)
        grid.fit(x_train, y_train)
        params = grid.best_params_
        logging.info(f'best parameters for svc: {params}')
        if params['kernel'] == 'rbf':
            return SVC(kernel = params['kernel'], C=params['C'], gamma= params['gamma'], probability=True)
        if params['kernel'] == 'linear':
            return  SVC(kernel = params['kernel'], C=params['C'], probability=True)

    def _find_best_tree(self,x_train, y_train, config):
        grid = GridSearchCV(estimator=tree.DecisionTreeClassifier(), param_grid=config, cv=5, scoring='accuracy',refit = True, verbose = 3)
        grid.fit(x_train, y_train)
        params = grid.best_params_
        logging.info(f'best parameters for decision trees: {params}')
        return tree.DecisionTreeClassifier(max_depth=params['max_depth'], 
                                           min_samples_leaf=params['min_samples_leaf'], 
                                           min_samples_split=params['min_samples_split'])

    def _find_best_forest(self,x_train, y_train, config):
        grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=config, cv=5, scoring='accuracy',refit = True, verbose = 3)
        grid.fit(x_train, y_train)
        params = grid.best_params_
        logging.info(f'best parameters for random forests: {params}')
        return RandomForestClassifier(max_depth=params['max_depth'], 
                                           min_samples_leaf=params['min_samples_leaf'], 
                                           min_samples_split=params['min_samples_split'],
                                           n_estimators=params['n_estimators'])

    def visualise_roc(self):
        X_train, X_test, y_train, y_test = self.train_test_data
        pred = self.predict(X_test=X_test)

        for model_name, pred in pred.items():
            fpr_tpr = utils.eer_for_visualisation(X_test, pred)
            fig = px.line(fpr_tpr, x='False Positive Rate', y='True Positive Rate', title = f'ROC Curve for {model_name}')
            fig.show()
    
    def plot_confusion_matrix(self):

        X_train, X_test, y_train, y_test = self.train_test_data
        fig ,ax = plt.subplots(figsize=(8,6))
        predictions = self.predict(X_test=X_test)
        for model_name, pred in predictions.items():
            cm = confusion_matrix(y_true=y_test, y_pred=pred, labels=['self', 'other', 'deepfake'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['self', 'other', 'deepfake'])
            disp.plot()
            plt.title(f'Confusion matrix for model {model_name}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()


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