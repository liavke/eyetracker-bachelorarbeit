import sys
import os
sys.path.append(os.getenv('PATH_TO_CLASSIFICATION'))

from classifiers import SVM_BaseClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import numpy as np

def visualise_auoc():
    pass

def calc_score():
    pass

def get_best_features_svm(X, y):
    score_df = pd.DataFrame()

    for feature_name in X:
        X_train, y_train, X_test, y_test = train_test_split(X[feature_name], y, test_size=0.2, shuffle=False)
        model = SVM_BaseClassifier()
        model.fit(X_train=X_train,y_train=y_train)
        score = model.evaluate()

        column_score_df = pd.DataFrame({
            "feature_name":[feature_name],
            "score": [score]
        })
        score_df = pd.concat([score_df,column_score_df])
        score_df.sort_values(by=['score'], ascending=False)

        """
        evaluation_list.append({
            "model": "SVM",
            "column names:": [n_columned_X.columns],
            "column number": [column_n],
            "score": score
        })
        """
    return score_df

def calculate_eer(y_test, predictions):
    """"
    This function calculates the average EER by calcualting the roc_curve for each label and then the EER for each label
    """

    #eer for self
    self_eer = eer(current_label='self', ground_truth=y_test, predictions=predictions)

    #eer for friend
    friend_eer = eer(current_label='other', ground_truth=y_test, predictions=predictions)
    
    #eer for deepfake
    deepfake_eer = eer(current_label='deepfake', ground_truth=y_test, predictions=predictions)

    avg_eer = (self_eer+friend_eer+deepfake_eer)/3
    return avg_eer


def eer(current_label, ground_truth, predictions):
     #source: https://stackoverflow.com/questions/28339746/equal-error-rate-in-python
    binary_ground_truth = [label == current_label for label in ground_truth]
    binary_prediction = [label == current_label for label in predictions]
    fpr, tpr, _ = roc_curve(binary_ground_truth, binary_prediction)
    fnr = 1 - tpr
    return fpr[np.nanargmin(np.absolute((fnr - fpr)))]