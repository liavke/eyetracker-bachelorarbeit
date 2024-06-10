import os
import pandas
from feature_extraction import FeatureExtractionPipeline
import pickle

def get_data_list(filepath: str) -> list[pandas.DataFrame]:
    data_list:list[pandas.DataFrame] = []
    for filename in os.listdir(filepath):
        file = os.path.join(filepath, filename)
        file_df = pandas.read_csv(file)
        #data = Data(table=file_df, columns=file_df.columns.columns.tolist)
        data_list.append(file_df)
    return data_list

""""
DATA PREPROCESSING:

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

def filter_columns(data: pandas.DataFrame):
    return data[['CNT', 'TIME', 'TIME_TICK', 'LPOGX', 'LPOGY', 'RPOGX', 'RPOGY', 'LPD',
       'LPS', 'LPV', 'RPD', 'RPS', 'RPV', 'RPUPILD', 'LPMM',
       'LPMMV', 'RPMM', 'RPMMV', 'USER','LABEL', 'GAZE_LABEL', 'BKDUR']]
       
def clear_blinks(data: pandas.DataFrame, extra:bool=False) -> pandas.DataFrame:
    """"
    This function remove any entries where the user blinks

    parameters:
        -data: The data entry, made up of a pandas Dataframe of eye tracker data and the columns
        -extra: a boolean variable in case the user would like to make sure any possible blink trial is dropped, 
        using the duartion of the blink

    result:
        filtered Data object without the blink data points
    """
    if extra:
        filtered_data = pandas.DataFrame()
        blink_dur_entries = data[data['BKDUR' ]==0]
        for _, blink_entry in  blink_dur_entries.iterrows():
            entry_df = data[(data['TIME']<=blink_entry['TIME'])&
                 (data['TIME']<=(blink_entry['TIME']+blink_entry["BKDUR"]))]
            filtered_data = pandas.concat([entry_df, filtered_data])
        return filtered_data.drop_duplicates().reset_index(drop=True)          
    return data[data['BKID']==0].reset_index(drop=False)

def handle_outliers(data: pandas.DataFrame) -> pandas.DataFrame:
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

def label_data(data: pandas.DataFrame):
    self_data = data.copy()
    self_data = self_data[(self_data['FPOGX']>=0.4) & (self_data['FPOGX']<=0.6)]
    self_data = self_data[(self_data['FPOGY']>=0.2) & (self_data['FPOGY'] <=0.6)]
    self_data['GAZE_LABEL'] = 'looking_at_self'

    not_self_data = data.copy()
    not_self_data = not_self_data[~not_self_data.index.isin(self_data.index)]    
    out = pandas.concat([self_data, not_self_data], sort=False).sort_index()

    out = out.fillna({'GAZE_LABEL': False})
    return out

def gaze_label_data(data: pandas.DataFrame):
    self_data = data.copy()
    stranger_data = data.copy()

    self_data = self_data[(self_data['FPOGX']>=0.4) & (self_data['FPOGX']<=0.6)]
    self_data = self_data[(self_data['FPOGY']>=0.2) & (self_data['FPOGY'] <=0.6)]
    self_data['GAZE_LABEL'] = 'looking_at_self'

    stranger_data = stranger_data[(stranger_data['FPOGX']>=0.8) & (stranger_data['FPOGX']<=1)]
    stranger_data = stranger_data[(stranger_data['FPOGY']>=0.2) & (stranger_data['FPOGY'] <=0.8)]
    stranger_data['GAZE_LABEL'] = 'looking_at_stranger'

    not_self_data = data[~data.index.isin(self_data.index)].copy()
    data_self = pandas.concat([self_data, not_self_data], sort=False)

    not_stranger_data = data[~data.index.isin(stranger_data.index)].copy()    
    data_stranger = pandas.concat([stranger_data, not_stranger_data], sort=False)

    out = pandas.concat([data_self, data_stranger], sort=False).sort_values(by=['TIME']).reset_index()
    
    out = out.fillna({'GAZE_LABEL': 'else'})
    return out
        

def calculate_featues(strategy='all', data:pandas.DataFrame = None):
    Y = data['LABEL'][0]
    data = data.drop(columns=['LABEL'])
    feature_pipeline = FeatureExtractionPipeline(data=data)
    feature_pipeline.run(strategy)
    X  = feature_pipeline.X
    return (X, Y)

def test_FEP(data):
    Y = data['LABEL'][0]
    data = data.drop(columns=['LABEL'])
    feature_pipeline = FeatureExtractionPipeline(data=data)
    feature_pipeline.standardise_data()
    feature_pipeline.get_dilation_periods()
    X  = feature_pipeline.X
    return (X, Y)

def save_data_as_pickle(data):
    with open('subject1.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)