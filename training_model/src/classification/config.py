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

class ClassifiersConfig():
    svc_config = [
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
    
    dt_config = param_grid = {
    'max_depth': [30, 50 ,70, 100],
    'min_samples_leaf': [5, 10, 15, 20],
    'min_samples_split': [10, 20, 30, 40],
    }

    #source: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    rf_config = {
    'max_depth': [30 ,50 ,70, 100],
    'min_samples_leaf': [5, 10, 15, 20],
    'min_samples_split': [10, 20, 30, 40],
    'n_estimators': [100, 400, 600, 1000]
}
    