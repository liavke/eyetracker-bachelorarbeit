import sys
import os
sys.path.append(os.getenv('PATH_TO_TM'))

import plotly.graph_objects as go 
import matplotlib.pyplot as plt

from src.dataset.dataset import Dataset
from src.classification.classifiers import BinaryBaseClassifiers

import src.dataset.utils as utils
import src.classification.utils as c_utils
import numpy as np
import pandas as pd

def label_visualization(ms_data): 
    fig = make_subplots(rows= 2, cols=2)
    options: list[str] = ['self', 'other', 'deepfake']

    for data in ms_data:

        start_time = data[0]['TIME'].iloc[0]
        end_time = data[0]['TIME'].iloc[-1]

        dilation_self = [entry['dilation'].values for entry in data if entry['LABEL'].values.all()=='self']
        dilation_other = [entry['dilation'].values for entry in data if entry['LABEL'].values.all()=='other']
        dilation_deepfake = [entry['dilation'].values for entry in data if entry['LABEL'].values.all()=='deepfake']
        
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
                #'difference between self/deepfake': self_df_diff_dil
        }

        for index, opt in enumerate(options):
                fig.add_trace(go.Scatter(
                    x=np.linspace(start=start_time, stop=end_time, num=(max_lengths[index])-6), 
                    y=plot_data[opt], 
                    mode='lines', name=opt,
                ))
    fig_title = 'Pupil size change over time for each labels of ' + data.subject
    fig.update_layout(
    title=fig_title,
    xaxis_title='Time, t in seconds',
    yaxis_title='Pupil size d, normalised'
    )
    fig.show()
    fig.to_html(f'mean_graph_{data.subject}.html')


def visualize_cm_from_raw():
     raw = pd.read_csv(filepath_or_buffer='/Users/liavkeren/bachelor_thesis/bachelor_thesis_code/training_model/src/data/results/subject3/raw_predictions_binary_1000ms.csv')
     c_utils.visualize_cm(predictions=raw.iloc[:,:-1], y_test=raw['ground truth'])

def main():
    current_subject = 'subject3'
    measurement_time = '1000ms'

    PATH =  os.getenv('PATH_TO_TM')+ "/src/data/" 
    SAVE_PATH = os.getenv('PATH_TO_TM')+ "/src/data/results/"
    dataset = Dataset(filepath=PATH, subject=current_subject)

    dataset.preprocess_data(measurement_timeframe=measurement_time)
    X, y = dataset.feature_extraction()

    new_y = [label == 'deepfake' for label in y]
    classifiers = BinaryBaseClassifiers(X=X, y=new_y)
    _,_ = classifiers.run()
    classifiers.visualise_roc(title=f'Roc curve for {current_subject}, at {measurement_time} measurement timeframe')

if __name__ == "__main__":
     main()