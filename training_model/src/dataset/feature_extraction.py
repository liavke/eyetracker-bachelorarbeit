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
    
    def get_dilation_periods(self, measurement_timeframe="POTSDAM_2023", k_window=5): #or windows? search a better name maybe
        """"
        Retrives the main measurement timeframe from when the user's gaze was directed at themselves. This timeframe is either defined by the user or defined by the research conducted by the Potsdam university (3000ms)[1]
        The data is then post processed to filter out any entries where the user does not look at themselves, and to handle blinking/missing data[2].

        Parameters:
        - measurement_timeframe: case options to either use user defined window size or the window size defined by the Potsdam research[1]
        - K_window: window size to be used if case is custom

        Return:
        - self.X: List of dictionaries that contain the rows in the interval calculated below

        Sources:
        [1] schwetlick-et-al_face-and-self-recognition.pdf
        [2]https://pandas.pydata.org/docs/reference/api/pandas.Series.bfill.html
        """
        temp_x = []
        else_df = self.X[self.X['GAZE_LABEL'] == 'else']
        print("#### EXTRACTING DILATION PERIOD DATA ####")

        match measurement_timeframe:
            case 'custom':
                for index, entry in self.X.iterrows():
                    if index == self.X.shape[0]-1:
                        break
                    next_entry = self.X.iloc[index+1]
                    if ((entry['GAZE_LABEL']in ['looking_at_stranger', 'looking_at_self'])):
                            if not self._has_blink(self.X.iloc[index+1: (index+k_window-1000)]):
                                temp_x.append(temp_data)
            
            case 'POTSDAM_2023':
                with alive_bar(len(self.X)) as bar:
                    for index, entry in else_df.iterrows():
                        time.sleep(0.1)
                        bar()

                        if index == else_df.shape[0]-1:
                            break
                        next_entry = self.X.iloc[index+1]
                        if (next_entry['GAZE_LABEL'] in ['looking_at_stranger', 'looking_at_self']):
                                
                                #find closest time entry to 3000ms from the start time https://www.statology.org/pandas-find-closest-value/
                                end_time_index = self.X.iloc[(self.X['TIME']-(next_entry['TIME']+3)).abs().argsort()[:1]].index[0]

                                #init the time
                                temp_data = self.X.iloc[index+1: end_time_index].copy()
                                temp_data['TIME'] = np.linspace(0, 3, temp_data.shape[0])

                                if not self._has_blink(temp_data.iloc[:-500]):
                                    temp_x.append(temp_data)

        #postprocessing
        self.X = self._post_processing(data=temp_x)

    def standardise_data(self):
        dilation_r = self.X[ColumnNames.DILATION_RIGHT]
        dilation_l = self.X[ColumnNames.DILATION_LEFT]
        self.X[ColumnNames.DILATION] = self._normalise_data(((dilation_r+dilation_l)/2).to_list())
        #todo: get only dilation, user and label column

    def _fill_missing_data(self, data):
        #source: https://www.kaggle.com/code/dansbecker/handling-missing-values
        my_imputer = SimpleImputer()
        data_with_imputed_values = my_imputer.fit_transform(data)
    
    def _normalise_data(self, data):
        xmin = min(data)
        xmax = max(data)
        return [(x - xmin) / (xmax - xmin) for x in data]
    
    def _post_processing(self, data):
        """"
        The data is post processed to filter out any entries where the user does not look at themselves.
        For missing data, backward filling is used. 

        ###FEW WORDS ABOUT BACKWARD FILLING###

        Parameters:
        - data

        Returns:
        - List of dictionaries that contain the rows in the dilation periods

        Sources:
        https://pandas.pydata.org/docs/reference/api/pandas.Series.bfill.html
        """
        X_new = []
        for data in (data):
            if data['looking_at_self'].eq(False).any():
                continue  
            backward_filled = data[ColumnNames.DILATION].replace(0, np.nan).bfill().to_numpy()
            data[ColumnNames.DILATION] = backward_filled
            X_new.append(data)
        return X_new
                            
    def _has_blink(self, data: pd.DataFrame) -> bool:
        """"
        This function checks if the dialtion period has any blinking in it

        parameters:
            -data: The data entry, made up of a pandas Dataframe of eye tracker data and the columns

        result:
            boolean
        """
        if data['BKDUR'].any() != 0:
            return True
        return False      