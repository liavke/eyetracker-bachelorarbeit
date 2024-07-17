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

import plotly.graph_objects as go

class Dataset():
    def __init__(self, filepath, subject) -> None:
        self.data: list[pd.DataFrame] = utils.get_data_list(filepath=filepath+subject)
        self.subject = subject
        self.filepath = filepath

    def preprocess_data(self, measurement_timeframe ="1000ms") -> list[pd.DataFrame]:
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
        print("### FEATURE EXTRACTION ###")
        X_df = pd.DataFrame()
        Y_list = []
        for entry in self.data:
            Y = entry['LABEL'].values[0]
            entry = entry.drop(columns=['LABEL'])
            feature_pipeline = FeatureExtractionPipeline(data=entry['dilation'].values)
            X = feature_pipeline.run()
            X_df = pd.concat([X_df, X])
            Y_list.append(Y)
        balanced_x, balanced_y = utils.balance_data(X=X_df, y=Y_list)
        return balanced_x, balanced_y

    def _get_dilation_periods(self, data, measurement_timeframe="3000ms"): #or windows? search a better name maybe
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

        #this is used to cut outliers datapoints that dont have suitable number of data (see holograms)
        cutoff_len = {
             "3000ms":400,
             "1500ms":200,
             "1000ms":120
        }

        with alive_bar(len(data)) as bar:
            for datapoint in data:
                time.sleep(0.0000000000000000000000000000000001)
                bar()
                else_indexes = datapoint[datapoint['GAZE_LABEL'] == 'else'].index
                self_indexes = datapoint[datapoint['GAZE_LABEL'] == 'gaze_self'].index
                other_indexes = datapoint[datapoint['GAZE_LABEL'] == 'gaze_other'].index
                print("#### EXTRACTING DILATION PERIOD DATA ####")

                match measurement_timeframe:
                    case '1000ms':
                            for index in else_indexes:
                                next_index = index+1
                                if next_index in self_indexes or next_index in other_indexes:
                                        start_time = datapoint.iloc[next_index]['TIME']

                                        #check if subject did not blink pior to gaze change
                                        second_priot_to_index = datapoint.iloc[(datapoint['TIME']-(start_time-1)).abs().argsort()[:1]].index[0]
                                        if not utils.has_blink(datapoint.iloc[second_priot_to_index:next_index]):
                                            
                                            #find closest time entry to 3000ms from the start time https://www.statology.org/pandas-find-closest-value/
                                            end_time_index = datapoint.iloc[(datapoint['TIME']-(start_time+1)).abs().argsort()[:1]].index[0]

                                            #init the time
                                            temp_data = datapoint.iloc[next_index: end_time_index].copy()
                                            temp_data['TIME'] = np.linspace(0, 1, temp_data.shape[0])

                                            if not utils.has_blink(temp_data):
                                                
                                                temp_x.append(temp_data)

                    case '1500ms':
                            for index in else_indexes:
                                    next_index = index+1
                                    if next_index in self_indexes or next_index in other_indexes:
                                            start_time = datapoint.iloc[next_index]['TIME']

                                            #check if subject did not blink pior to gaze change
                                            second_priot_to_index = datapoint.iloc[(datapoint['TIME']-(start_time-1)).abs().argsort()[:1]].index[0]
                                            if not utils.has_blink(datapoint.iloc[second_priot_to_index:next_index]):
                                                
                                                #find closest time entry to 1500ms from the start time https://www.statology.org/pandas-find-closest-value/
                                                end_time_index = datapoint.iloc[(datapoint['TIME']-(start_time+1.5)).abs().argsort()[:1]].index[0]

                                                #init the time
                                                temp_data = datapoint.iloc[next_index: end_time_index].copy()
                                                temp_data['TIME'] = np.linspace(0, 1.5, temp_data.shape[0])

                                                if not utils.has_blink(temp_data.iloc[:-500]):
                                                    
                                                    temp_x.append(temp_data)
                    
                    case '3000ms':
                            for index in else_indexes:
                                next_index = index+1
                                if next_index in self_indexes or next_index in other_indexes:
                                        start_time = datapoint.iloc[next_index]['TIME']

                                        #check if subject did not blink pior to gaze change
                                        second_priot_to_index = datapoint.iloc[(datapoint['TIME']-(start_time-1)).abs().argsort()[:1]].index[0]
                                        if not utils.has_blink(datapoint.iloc[index:next_index]):
 
                                            #find closest time entry to 3000ms from the start time https://www.statology.org/pandas-find-closest-value/
                                            end_time_index = datapoint.iloc[(datapoint['TIME']-(start_time+3)).abs().argsort()[:1]].index[0]

                                            #init the time
                                            temp_data = datapoint.iloc[next_index: end_time_index].copy()
                                            #temp_data['TIME'] = np.linspace(0, 3, temp_data.shape[0])

                                            if not utils.has_blink(temp_data.iloc[:-500]):
                                                
                                                temp_x.append(temp_data)
        return self._post_processing(data=temp_x, min_length=cutoff_len[measurement_timeframe])

        #postprocessing
    
    def _standardise_data(self, datapoint):
        dilation = (datapoint[ColumnNames.DILATION_RIGHT] + datapoint[ColumnNames.DILATION_LEFT])/2
        dilation = utils.normalise_data(dilation)
        dilation = utils.smoothing(window_size=5, strategy='gaussian', data=dilation.values)
        #dilation = utils.baseline(dilation)
        datapoint[ColumnNames.DILATION] = dilation
        
        return datapoint
        #todo: get only dilation, user and label column

    def _post_processing(self, data, min_length):
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
        outlier = []
        for entry in (data):
            if entry['GAZE_LABEL'].eq('else').any():
                continue  
            backward_filled = entry[ColumnNames.DILATION].replace(0, np.nan).bfill().to_numpy()
            if len(backward_filled)>10:#min_length:
                baseline_corrected = utils.baseline(backward_filled)
                entry[ColumnNames.DILATION] = baseline_corrected
                X_new.append(entry)
            outlier.append(backward_filled)
        return X_new   
    
    def _gaze_label_data(self, data: pd.DataFrame):
        self_data = data.copy()
        stranger_data = data.copy()

        self_data = self_data[(self_data['FPOGX']>=0.4) & (self_data['FPOGX']<=0.6)]
        self_data = self_data[(self_data['FPOGY']>=0.2) & (self_data['FPOGY'] <=0.8)]
        self_data['GAZE_LABEL'] = 'gaze_self'

        stranger_data = stranger_data[(stranger_data['FPOGX']>=0.8) & (stranger_data['FPOGX']<=1)]
        stranger_data = stranger_data[(stranger_data['FPOGY']>=0.2) & (stranger_data['FPOGY'] <=0.8)]
        stranger_data['GAZE_LABEL'] = 'gaze_other'

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
            if entry['GAZE_LABEL'].eq('gaze_other').all():
                entry['LABEL'] = 'other'
                out.append(entry[['dilation','LABEL', 'TIME']])
            else:
                out.append(entry[['dilation','LABEL','TIME']])
        return out
    
    def _set_subject(self, data):
        data['USER'] = self.subject
        return data
    
    def visualisation(self, options: list[str] = ['self', 'other', 'deepfake', 'difference between self/deepfake']):
        fig = go.Figure()

        dilation_self = [entry['dilation'].values for entry in self.data if entry['LABEL'].values.all()=='self']
        dilation_other = [entry['dilation'].values for entry in self.data if entry['LABEL'].values.all()=='other']
        dilation_deepfake = [entry['dilation'].values for entry in self.data if entry['LABEL'].values.all()=='deepfake']
        
        max_lengths = [max(len(entry) for entry in dilation_self),
                        max(len(entry) for entry in dilation_other),
                        max(len(entry) for entry in dilation_deepfake)]
        
        filtered_dil_self = utils.fill_missing_data_with_nan(dilation=dilation_self, max_length=max_lengths[0])
        filtered_dil_other = utils.fill_missing_data_with_nan(dilation=dilation_other, max_length=max_lengths[1])
        filtered_dil_deepfake = utils.fill_missing_data_with_nan(dilation=dilation_deepfake, max_length=max_lengths[2])
         
       
        #only a few data points are of max length, this results in very steep changes in the graph. Taking thse points away, cleans the graph
        mean_dilations = [
             np.nanmean(filtered_dil_self, axis=0)[:-6],
             np.nanmean(filtered_dil_other, axis=0)[:-6],
             np.nanmean(filtered_dil_deepfake, axis=0)[:-6]
        ]

        diff_len = np.min([len(mean_dilations[2]), len(mean_dilations[0])])
        self_df_diff_dil = mean_dilations[2][:diff_len]-mean_dilations[0][:diff_len]
        max_lengths.append(diff_len)

        plot_data = {
             'self' : mean_dilations[0],
             'other' : mean_dilations[1],
             'deepfake': mean_dilations[2],
             'difference between self/deepfake': self_df_diff_dil
        }

        for index, opt in enumerate(options):
              fig.add_trace(go.Scatter(
                   x=np.linspace(start=0, stop=3, num=(max_lengths[index])-6), 
                   y=plot_data[opt], 
                   mode='lines', name=opt,
              ))
        fig_title = 'Pupil size change over time for each labels of ' + self.subject
        fig.update_layout(
    title=fig_title,
    xaxis_title='Time, t in seconds',
    yaxis_title='Pupil size d, normalised'
)
        fig.show()
        fig.to_html('mean_graph_sub2.html')

    def visualize_raw_data(self, options):
        fig = go.Figure()

        dilation_self = [entry['dilation'].values for entry in self.data if entry['LABEL'].values.all()=='self']
        dilation_other = [entry['dilation'].values for entry in self.data if entry['LABEL'].values.all()=='other']
        dilation_deepfake = [entry['dilation'].values for entry in self.data if entry['LABEL'].values.all()=='deepfake']

        dilation_data = {
             'self': dilation_self,
             'other': dilation_other,
             'deepfake':dilation_deepfake
        }

        min_lengths = [min(len(entry) for entry in dilation_self),
                        min(len(entry) for entry in dilation_other),
                        min(len(entry) for entry in dilation_deepfake)]

        for index, opt in enumerate(options):
              for entry in dilation_data[opt]:
                fig.add_trace(go.Scatter(
                    x=np.linspace(start=0, stop=3, num=min_lengths[index]), 
                    y=entry, 
                    mode='lines'
                ))
        fig_title = 'Graph of labels for ' + self.subject
        fig.update_layout(
             title=fig_title
        )
        fig.show()

    def plot_dilation_over_time(self):
        fig = go.Figure()

        self_ = [entry for entry in self.data if entry['LABEL'].values.all()=='self']
        deepfake_ = [entry for entry in self.data if entry['LABEL'].values.all()=='deepfake']

        self_dilation= []
        self_time=[]

        deepfake_dilation =[]
        deepfake_time =[]


        for d in self_:
            self_dilation.extend(d['dilation'].values)
            self_time.extend(d['TIME'].values)

        for d in deepfake_:
            deepfake_dilation.extend(d['dilation'].values)
            deepfake_time.extend(d['TIME'].values)

        fig.add_trace(go.Scatter(
                    x=np.linspace(start=0, stop=len(self_dilation)), 
                    y=self_dilation, 
                    mode='lines', name='self'
                ))
        fig.add_trace(go.Scatter(
                    x=np.linspace(start=0, stop=len(deepfake_dilation)), 
                    y=deepfake_dilation, 
                    mode='lines', name= 'deepfake'
                ))
        
        fig.show()
        

