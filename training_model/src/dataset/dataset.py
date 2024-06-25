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

    def preprocess_data(self, measurement_timeframe ="3000ms") -> list[pd.DataFrame]:
        print("#### PREPROCESSING DATA ####")
        processed_data = []
        for datapoint in self.data:
                
                #datapoint = datapoint.dropna(how='any')
                if datapoint.empty:
                    continue

                #datapoint = utils.clear_blinks(datapoint)
                datapoint = self._set_subject(datapoint)
                datapoint = utils.handle_outliers(datapoint)
                datapoint = self._gaze_label_data(datapoint)
                datapoint = utils.filter_columns(datapoint)
                datapoint = self._standardise_data(datapoint)
                processed_data.append(datapoint)
               
        dilation_data = self._get_dilation_periods(processed_data, measurement_timeframe=measurement_timeframe)
        dilation_data = self._rename_columns(dilation_data)
                #datapoint.dropna()
                
        self.data = dilation_data
        #self._save_data_as_pkl()

    def feature_extraction(self):
        #data = utils.standardise_data(self.data)
        X_df = pd.DataFrame()
        Y_list = []
        for entry in self.data:
            Y = entry['LABEL'].values[0]
            entry = entry.drop(columns=['LABEL'])
            feature_pipeline = FeatureExtractionPipeline(data=entry['dilation'].values)
            X = feature_pipeline.run()
            X_df = pd.concat([X_df, X])
            Y_list.append(Y)
        return X_df, Y_list

    def _get_dilation_periods(self, data, measurement_timeframe="1500ms"): #or windows? search a better name maybe
        """"
        Retrives the main measurement timeframe from when the user's gaze was directed at themselves. This timeframe is either defined by the user or defined by the research conducted by the Potsdam university (3000ms)[1]
        The data is then post processed to filter out any entries where the user does not look at themselves, and to handle blinking/missing data[2].

        Parameters:
        - measurement_timeframe: case options to either use user defined defined by the Potsdam research[1] or by ICJB 2023[3]
        - K_window: window size to be used if case is custom

        Return:
        - datapoint: List of dictionaries that contain the rows in the interval calculated below

        Sources:
        [1] schwetlick-et-al_face-and-self-recognition.pdf
        [2]https://pandas.pydata.org/docs/reference/api/pandas.Series.bfill.html
        [3]
        """
        temp_x = []
        with alive_bar(len(data)) as bar:
            for datapoint in data:
                time.sleep(0.0000000000000000000000000000000001)
                bar()
                else_indexes = datapoint[datapoint['GAZE_LABEL'] == 'else'].index
                self_indexes = datapoint[datapoint['GAZE_LABEL'] == 'looking_at_self'].index
                stranger_indexes = datapoint[datapoint['GAZE_LABEL'] == 'looking_at_familiars'].index
                print("#### EXTRACTING DILATION PERIOD DATA ####")

                match measurement_timeframe:
                    case '1500ms':
                        for index in else_indexes:
                                next_index = index+1
                                if next_index in self_indexes or next_index in stranger_indexes:
                                        start_time = datapoint.iloc[next_index]['TIME']
                                        
                                        #find closest time entry to 1500ms from the start time https://www.statology.org/pandas-find-closest-value/
                                        end_time_index = datapoint.iloc[(datapoint['TIME']-(start_time+1.5)).abs().argsort()[:1]].index[0]

                                        #init the time
                                        temp_data = datapoint.iloc[next_index: end_time_index].copy()
                                        temp_data['TIME'] = np.linspace(0, 3, temp_data.shape[0])

                                        if not utils.has_blink(temp_data.iloc[:-500]):
                                            
                                            temp_x.append(temp_data)
                    
                    case '3000ms':
                            for index in else_indexes:
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
        return self._post_processing(data=temp_x)
    
    def _standardise_data(self, datapoint):
        dilation = (datapoint[ColumnNames.DILATION_RIGHT] + datapoint[ColumnNames.DILATION_LEFT])/2
        dilation = utils.smoothing(window_size=5, strategy='gaussian', data=dilation.values)
        dilation = utils.normalise_data(dilation)
        dilation = utils.baseline(dilation)
        datapoint[ColumnNames.DILATION] = dilation
        
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
            if data['GAZE_LABEL'].eq('else').any():
                continue  
            if len(data) >= 400:
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
        stranger_data['GAZE_LABEL'] = 'looking_at_familiars'

        gaze_data = pd.concat([self_data, stranger_data], sort=False)

        not_gaze_data = data[~data.index.isin(gaze_data.index)].copy()    

        out = pd.concat([gaze_data, not_gaze_data], sort=False).sort_values(by=['TIME']).reset_index()
        
        out = out.fillna({'GAZE_LABEL': 'else'})
        return out
    
    def _save_data_as_pkl(self):
        with open(self.filepath+'/'+self.subject+'.pkl', 'wb') as file:
            pickle.dump(self.data, file)

    def _rename_columns(self, data):
        out = []
        for entry in data:
            if(type(data))==None:
                print("None data")
            if entry['GAZE_LABEL'].eq('looking_at_familiars').all():
                entry['LABEL'] = 'friend'
                out.append(entry[['dilation','LABEL', 'TIME']])
            else:
                out.append(entry[['dilation','LABEL','TIME']])
        return out
    
    def _set_subject(self, data):
        data['USER'] = self.subject
        return data