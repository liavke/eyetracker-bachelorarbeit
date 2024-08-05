import pandas as pd
from config import  Features
import numpy as np
import numpy as np
from alive_progress import alive_bar
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
                Features.VAR: [np.var(self.X)],
            })