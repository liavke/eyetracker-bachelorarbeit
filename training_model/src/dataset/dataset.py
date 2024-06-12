import sys
import os
sys.path.append(os.getenv('PATH_TO_TM_DATASET'))

import utils
from config import Data
import pandas as pd
from alive_progress import alive_bar
import time
from config import ColumnNames
import numpy as np
from feature_extraction import FeatureExtractionPipeline
import pickle 

class Dataset():
    def __init__(self, filepath, subject) -> None:
        self.data: list[pd.DataFrame] = utils.get_data_list(filepath=filepath)
        self.subject = subject
        self.filepath = filepath

    def preprocess_data(self, measurement_timeframe ="POTSDAM_2023") -> list[pd.DataFrame]:
        print("#### PREPROCESSING DATA ####")
        processed_data = []
        for datapoint in self.data:
                
                #datapoint = datapoint.dropna(how='any')
                if datapoint.empty:
                    continue

                #datapoint = utils.clear_blinks(datapoint)
                datapoint = utils.handle_outliers(datapoint)
                datapoint = self._gaze_label_data(datapoint)
                datapoint = utils.filter_columns(datapoint)
                datapoint = self._standardise_data(datapoint)
                datapoint = self._get_dilation_periods(datapoint, measurement_timeframe=measurement_timeframe, k_window=5)
                #datapoint.dropna()
                processed_data.append(datapoint)
        self.data = processed_data

    def feature_extraction(self, strategy):
        #data = utils.standardise_data(self.data)
        X_df = pd.DataFrame()
        Y_list = []
        for entry in self.data:
            Y = data['LABEL'][0]
            data = data.drop(columns=['LABEL'])
            feature_pipeline = FeatureExtractionPipeline(data=entry)
            feature_pipeline.run(strategy)
            X  = feature_pipeline.X
            X_df = pd.concat([X_df, X])
            Y_list.append(Y)
        return X_df, Y_list

    def _get_dilation_periods(self, datapoint, measurement_timeframe="POTSDAM_2023", k_window=5): #or windows? search a better name maybe
        """"
        Retrives the main measurement timeframe from when the user's gaze was directed at themselves. This timeframe is either defined by the user or defined by the research conducted by the Potsdam university (3000ms)[1]
        The data is then post processed to filter out any entries where the user does not look at themselves, and to handle blinking/missing data[2].

        Parameters:
        - measurement_timeframe: case options to either use user defined window size or the window size defined by the Potsdam research[1]
        - K_window: window size to be used if case is custom

        Return:
        - datapoint: List of dictionaries that contain the rows in the interval calculated below

        Sources:
        [1] schwetlick-et-al_face-and-self-recognition.pdf
        [2]https://pandas.pydata.org/docs/reference/api/pandas.Series.bfill.html
        """
        temp_x = []
        else_indexes = datapoint[datapoint['GAZE_LABEL'] == 'else'].index
        self_indexes = datapoint[datapoint['GAZE_LABEL'] == 'looking_at_self'].index
        stranger_indexes = datapoint[datapoint['GAZE_LABEL'] == 'looking_at_stranger'].index
        print("#### EXTRACTING DILATION PERIOD DATA ####")

        match measurement_timeframe:
            case 'custom':
                for index, entry in datapoint.iterrows():
                    if index == datapoint.shape[0]-1:
                        break
                    next_entry = datapoint.iloc[index+1]
                    if ((entry['GAZE_LABEL']in ['looking_at_stranger', 'looking_at_self'])):
                            if not utils.has_blink(datapoint.iloc[index+1: (index+k_window-1000)]):
                                temp_x.append(temp_data)
            
            case 'POTSDAM_2023':
                with alive_bar(len(datapoint)) as bar:
                    for index in else_indexes:
                        time.sleep(0.1)
                        bar()

                        next_index = index+1
                        if next_index in self_indexes or next_index in stranger_indexes:
                                start_time = datapoint.iloc[next_index]['TIME']
                                
                                #find closest time entry to 3000ms from the start time https://www.statology.org/pandas-find-closest-value/
                                end_time_index = datapoint.iloc[(datapoint['TIME']-(start_time+3)).abs().argsort()[:1]].index[0]

                                #init the time
                                temp_data = datapoint.iloc[next_index: end_time_index].copy()
                                temp_data['TIME'] = np.linspace(0, 3, temp_data.shape[0])

                                if not utils.has_blink(temp_data.iloc[:-500]):
                                    temp_x.append(temp_data)

        #postprocessing
        datapoint = self._post_processing(data=temp_x)
    
    def _standardise_data(self, datapoint):
        dilation_r = datapoint[ColumnNames.DILATION_RIGHT]
        dilation_l = datapoint[ColumnNames.DILATION_LEFT]
        datapoint[ColumnNames.DILATION] = utils.normalise_data(((dilation_r+dilation_l)/2).to_list())
        return datapoint
        #todo: get only dilation, user and label column

    def _post_processing(self, data):
        """"
        The data is post processed to filter out any entries where the user does not look at themselves or at a stranger.
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
            if data['GAZE_LABEL'].eq(False).any() or data['GAZE_LABEL'].eq('else').any():
                continue  
            backward_filled = data[ColumnNames.DILATION].replace(0, np.nan).bfill().to_numpy()
            data[ColumnNames.DILATION] = backward_filled
            X_new.append(data)
        return X_new   
    
    def _gaze_label_data(self, data: pd.DataFrame):
        self_data = data.copy()
        stranger_data = data.copy()

        self_data = self_data[(self_data['FPOGX']>=0.4) & (self_data['FPOGX']<=0.6)]
        self_data = self_data[(self_data['FPOGY']>=0.2) & (self_data['FPOGY'] <=0.6)]
        self_data['GAZE_LABEL'] = 'looking_at_self'

        stranger_data = stranger_data[(stranger_data['FPOGX']>=0.8) & (stranger_data['FPOGX']<=1)]
        stranger_data = stranger_data[(stranger_data['FPOGY']>=0.2) & (stranger_data['FPOGY'] <=0.8)]
        stranger_data['GAZE_LABEL'] = 'looking_at_stranger'

        gaze_data = pd.concat([self_data, stranger_data], sort=False)

        not_gaze_data = data[~data.index.isin(gaze_data.index)].copy()    

        out = pd.concat([gaze_data, not_gaze_data], sort=False).sort_values(by=['TIME']).reset_index()
        
        out = out.fillna({'GAZE_LABEL': 'else'})
        return out
    
    def _save_data_as_pkl():
        with open({self.subject}+'.pkl', 'wb') as file:
            # Write the data to the file
            pickle.dump(data, file)