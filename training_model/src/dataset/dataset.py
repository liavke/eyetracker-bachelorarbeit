import sys
import os
sys.path.append(os.getenv('PATH_TO_TM_DATASET'))

import utils
from config import Data
import pandas as pd

class Dataset():
    def __init__(self, filepath) -> None:
        self.data: list[pd.DataFrame] = utils.get_data_list(filepath=filepath)

    def preprocess_data(self) -> list[pd.DataFrame]:
        processed_data = []
        for datapoint in self.data:

            #datapoint = datapoint.dropna(how='any')
            if datapoint.empty:
                continue

            datapoint = utils.clear_blinks(datapoint)
            datapoint = utils.handle_outliers(datapoint)
            datapoint = utils.label_data(datapoint)
            datapoint = utils.filter_columns(datapoint)
            datapoint.dropna()
            processed_data.append(datapoint)
        self.data = processed_data
    
    def feature_extraction(self):
        #data = utils.standardise_data(self.data)
        X_df = pd.DataFrame()
        Y_list = []
        for entry in self.data:
            X, Y = utils.calculate_featues(strategy = 'statistical', data= entry)
            X_df = pd.concat([X_df, X])
            Y_list.append(Y)
        return X_df, Y_list
    
    def test_feature_extraction(self):
       X_list = []
       Y_list = []
       for entry in self.data:
           X,Y =  utils.test_FEP(data=entry)
           X_list.append(X)
           Y_list.append(Y)
       return (X_list,Y_list)