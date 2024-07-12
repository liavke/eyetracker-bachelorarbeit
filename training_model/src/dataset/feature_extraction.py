import pandas as pd
from config import ColumnNames, Features
import numpy as np
from sklearn.impute import SimpleImputer
import numpy as np
from alive_progress import alive_bar
import time
from scipy.fft import fft
pd.options.mode.chained_assignment = None  # default='warn'

class FeatureExtractionPipeline:
    def __init__(self, data) -> None:
        self.X:pd.DataFrame = data
    
    def run(self):
        """"
        Calculates statistical features (mean, max, min, stardard diviation, variance) for each self.X period in the data.

        Parameters: 
        - data: list of dictionaries of the dilaiton periods

        Return:
        - data: dictionary of the statistical features with each row representing these features for each self.X period

        Sources:
        - https://www.kaggle.com/code/pmarcelino/data-analysis-and-feature-extraction-with-python
        """
        ft = fft(self.X)
        S = np.abs(ft ** 2) / len(self.X)
        return pd.DataFrame({
                Features.MAX: [np.max(self.X)],
                Features.MEAN: [np.mean(self.X)],
                Features.MIN: [np.min(self.X)],
                Features.STD: [np.std(self.X)],
                #Features.VAR: [np.var(self.X)],
                #Features.POWER : [np.mean(self.X ** 2)],
                #Features.PEAK : [np.max(np.abs(self.X))],
                #Features.P2P : [np.ptp(self.X)],
                #Features.CRESTFACTOR :[np.max(np.abs(self.X)) / np.sqrt(np.mean(self.X ** 2))],
                Features.MAX_FOURIER: np.max(S),
                #Features.SUM_FOURIER: np.sum(S),
                Features.MEAN_FOURIER: np.mean(S),
                Features.VAR_FOURIER: np.var(S),
                Features.PEAK_FOURIER: np.max(np.abs(S)),
            })