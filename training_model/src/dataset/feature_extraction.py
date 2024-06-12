import pandas as pd
from config import ColumnNames, Features
import numpy as np
from sklearn.impute import SimpleImputer
import numpy as np
from alive_progress import alive_bar
import time
pd.options.mode.chained_assignment = None  # default='warn'

class FeatureExtractionPipeline:
    def __init__(self, data) -> None:
        self.X:pd.DataFrame = data
    
    def run(self, strategy, measurement_timeframe:str = 'POTSDAM_2023'):
        self.standardise_data()
        self.get_dilation_periods( measurement_timeframe=measurement_timeframe, k_window=5)
        #self.calculate_statistical()
    
    def calculate_statistical(self):
        """"
        Calculates statistical features (mean, max, min, stardard diviation, variance) for each dilation period in the data.

        Parameters: 
        - data: list of dictionaries of the dilaiton periods

        Return:
        - data: dictionary of the statistical features with each row representing these features for each dilation period

        Sources:
        - https://www.kaggle.com/code/pmarcelino/data-analysis-and-feature-extraction-with-python
        """
        features_df = pd.DataFrame
        for  dp in self.X:
            dilation = dp[ColumnNames.DILATION].values
            temp_data = {
                    Features.MAX: [np.max(dilation)],
                    Features.MEAN: [np.mean(dilation)],
                    Features.MIN: [np.min(dilation)],
                    Features.STD: [np.std(dilation)],
                    Features.VAR: [np.var(dilation)],
                }
            features_df = pd.concat(features_df, temp_data)
            

        self.X = features_df
    
    def calculate_fourier(self, data):
        return data      