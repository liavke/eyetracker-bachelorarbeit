from classifiers import SVM_Classifier, Decision_Trees_Classifier, K_Means_Classifier, Naive_Bayes_Classifier
import pandas as pd
from sklearn.model_selection import train_test_split

def visualise_auoc():
    pass

def cal_eer():
    pass

def calc_score():
    pass

def get_best_features_svm(X, y):
    score_df = pd.DataFrame()

    for feature_name in X:
        X_train, y_train, X_test, y_test = train_test_split(X[feature_name], y, test_size=0.2, shuffle=False)
        model = SVM_Classifier()
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