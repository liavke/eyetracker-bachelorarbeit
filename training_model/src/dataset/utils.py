import os
import pandas as pd
from feature_extraction import FeatureExtractionPipeline
import pickle
import numpy as np

from scipy.ndimage import gaussian_filter1d

def get_data_list(filepath: str) -> list[pd.DataFrame]:
    data_list:list[pd.DataFrame] = []
    for filename in os.listdir(filepath):
        if filename.endswith('.pkl'):
            pkl_data =  get_pkl_data(os.path.join(filepath, filename))
            data_list.append(pkl_data) #not sure if it should append it
            return data_list
        if filename.endswith('.csv'):
            file = os.path.join(filepath, filename)
            file_df = pd.read_csv(file)
            #data = Data(table=file_df, columns=file_df.columns.columns.tolist)
            data_list.append(file_df)
    return data_list

def get_pkl_data(filepath):
     with open(filepath, 'rb') as f:
        return pickle.load(f)

def filter_columns(data: pd.DataFrame):
    """"
    To see information about what each columns mean, see:

    1)  Prarthana Pillai, Prathamesh Ayare, Balakumar Balasingam, Kevin Milne, Francesco Biondi,
        Response time and eye tracking datasets for activities demanding varying cognitive load,
        Data in Brief,
        Volume 33,
        2020,
        106389,
        ISSN 2352-3409,
        https://doi.org/10.1016/j.dib.2020.106389.

    2)  https://www.gazept.com/dl/Gazepoint_API_v2.0.pdf
    """
    return data[['CNT', 'TIME', 'TIME_TICK', 'LPOGX', 'LPOGY', 'RPOGX', 'RPOGY', 'LPD',
       'LPS', 'LPV', 'RPD', 'RPS', 'RPV', 'RPUPILD', 'LPMM',
       'LPMMV', 'RPMM', 'RPMMV', 'USER','LABEL', 'GAZE_LABEL', 'BKDUR']]
       
def clear_blinks(data: pd.DataFrame, extra:bool=False) -> pd.DataFrame:
    """"
    This function remove any entries where the user blinks

    parameters:
        -data: The data entry, made up of a pd Dataframe of eye tracker data and the columns
        -extra: a boolean variable in case the user would like to make sure any possible blink trial is dropped, 
        using the duartion of the blink

    result:
        filtered Data object without the blink data points
    """
    if extra:
        filtered_data = pd.DataFrame()
        blink_dur_entries = data[data['BKDUR' ]==0]
        for _, blink_entry in  blink_dur_entries.iterrows():
            entry_df = data[(data['TIME']<=blink_entry['TIME'])&
                 (data['TIME']<=(blink_entry['TIME']+blink_entry["BKDUR"]))]
            filtered_data = pd.concat([entry_df, filtered_data])
        return filtered_data.drop_duplicates().reset_index(drop=True)          
    return data[data['BKID']==0].reset_index(drop=False)

def handle_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """"
    Function to filter out data outliers such as point of gaze outside the screen

    parameters:
    -data: data point of type DataFrame
    """
    out = data.copy()
    #handle point of gaze outliers
    out = out[(out['FPOGX']>=0) & (out['FPOGX']<1)]
    out = out[(out['FPOGY']>=0) & (out['FPOGY'] < 1)]
    out = out[(out['BPOGX']>=0) & (out['BPOGX']<1)]
    out = out[(out['BPOGY']>=0) & (out['BPOGY'] < 1)]
    return out.reset_index(drop=False)

def label_data(data: pd.DataFrame):
    self_data = data.copy()
    self_data = self_data[(self_data['FPOGX']>=0.4) & (self_data['FPOGX']<=0.6)]
    self_data = self_data[(self_data['FPOGY']>=0.2) & (self_data['FPOGY'] <=0.6)]
    self_data['GAZE_LABEL'] = 'looking_at_self'

    not_self_data = data.copy()
    not_self_data = not_self_data[~not_self_data.index.isin(self_data.index)]    
    out = pd.concat([self_data, not_self_data], sort=False).sort_index()

    out = out.fillna({'GAZE_LABEL': False})
    return out

def save_data_as_pickle(data):
    with open('subject1.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def normalise_data(data):
        xmin = np.min(data)
        xmax = np.max(data)
        return (data - xmin) / (xmax - xmin) 

def has_blink(data: pd.DataFrame) -> bool:
        """"
        This function checks if the dialtion period has any blinking in it

        parameters:
            -data: The data entry, made up of a pd Dataframe of eye tracker data and the columns

        result:
            boolean
        """
        if data['BKDUR'].any() != 0:
            return True
        return False  

def baseline(data):
     return data-data[0]


def smoothing(strategy, window_size, data):
    #todo: supply source
    match strategy:
        case 'convolution':
            window = np.ones(window_size) / window_size
            return np.convolve(data, window, mode='valid')
        
        case 'gaussian':
            sigma = 1
            return gaussian_filter1d(data, sigma)