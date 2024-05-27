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
            datapoint = utils.filter_columns(datapoint)
            processed_data.append(datapoint)
        return processed_data
    
    def feature_extraction(self):
        data = utils.standardise_data(self.data)
        X, Y = utils.calculate_features(feature_type = 'statistical', data= data)