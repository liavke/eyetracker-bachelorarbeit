import sys
import os
sys.path.append(os.getenv('PATH_TO_CLASSIFICATION'))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import numpy as np
from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def visualise_auoc():
    pass

def calc_score():
    pass

def get_best_features_svm(X, y):
    score_df = pd.DataFrame()

    for feature_name in X:
        X_train, y_train, X_test, y_test = train_test_split(X[feature_name], y, test_size=0.2, shuffle=False)
        model = None
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
    binary_ground_truth, binary_prediction = turn_data_binary(current_label='self', ground_truth=y_test, predictions=predictions)
    self_eer  = eer( binary_ground_truth, binary_prediction)

    #eer for friend
    binary_ground_truth, binary_prediction = turn_data_binary(current_label='other', ground_truth=y_test, predictions=predictions)
    other_eer = eer( binary_ground_truth, binary_prediction)
    
    #eer for deepfake
    binary_ground_truth, binary_prediction = turn_data_binary(current_label='deepfake', ground_truth=y_test, predictions=predictions)
    deepfake_eer  = eer( binary_ground_truth, binary_prediction)

    avg_eer = (self_eer+other_eer+deepfake_eer)/3
    return avg_eer

def turn_data_binary(current_label, ground_truth, predictions):
    binary_ground_truth = [label == current_label for label in ground_truth]
    binary_prediction = [label == current_label for label in predictions]
    return binary_ground_truth, binary_prediction


def eer( ground_truth, predictions):
     #source: https://stackoverflow.com/questions/28339746/equal-error-rate-in-python
    fpr, tpr, _ = roc_curve(ground_truth, predictions)
    fnr = 1 - tpr
    return fpr[np.nanargmin(np.absolute((fnr - fpr)))]

def calculate_percision_recall(y_true, predictions):
    #source: https://stackoverflow.com/questions/52215603/how-to-get-precision-and-recall-using-linear-svc-of-svm 
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    precision_score = tp / (tp + fp)
    recall_score = tp / (tp + fn)
    return precision_score, recall_score

def eer_for_visualisation(ground_truth, predictions):
    fpr, tpr, _ = roc_curve(ground_truth, predictions)
    return pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate':tpr})

def calculate_average_aucroc(y_train,y_test, predictions):
    #source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)

def visualize_cm(predictions, y_test):
    for model_name, pred in predictions.items():
        print(model_name)
        print(f'accuracy: {accuracy_score(y_pred=pred, y_true=y_test)}')
        print(f'eer: {eer(ground_truth=y_test, predictions=pred)}')

        cm = confusion_matrix(y_true=y_test, y_pred=pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion matrix for model {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

def visualise_roc(y_test, pred, title):
    fig = go.Figure()

    for model_name, pred in pred.items():
        fpr, tpr, _ = roc_curve(y_test, pred[:,1])
        fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines', name=model_name
            ))
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )
    fig.show()