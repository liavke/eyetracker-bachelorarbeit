import sys
import os
sys.path.append(os.getenv('PATH_TO_TM'))
from src.dataset.dataset import Dataset
import unittest

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from collections import Counter
from src.classification.classifiers import MultiBaseClassifiers

PATH =  os.getenv('PATH_TO_TM')+ "/src/data/"

def test_data_loader():
    pass

def test_preprocessing():
    dataset = Dataset(filepath=PATH, subject='subject2')
    dataset.preprocess_data()
    return dataset.data

def test_feature_extraction():
    dataset = Dataset(filepath=PATH, subject='subject2')
    dataset.preprocess_data(measurement_timeframe="3000ms")
    X, y = dataset.feature_extraction()
    counter = Counter(y)
    self_count = (counter['self'])
    deepfake_count = (counter['deepfake'])
    friend_count = (counter['other'])
    print(f'count for self: {self_count}')
    print(f'count for deepfake: {deepfake_count}')
    print(f'count for other: {friend_count}')


class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

def test_visualizing_data():
    dataset = Dataset(filepath=PATH, subject='subject3')
    dataset.preprocess_data(measurement_timeframe="3000ms")
    dataset.visualisation()
    #ataset.plot_dilation_over_time(label='self')

def get_time_per_subject():
    deepfake_time = 0
    self_time = 0
    dataset = Dataset(filepath=PATH, subject='subject2')

    for entry in dataset.data:
        time = entry['TIME'].values
        if 'self' in entry['LABEL'].values:
            self_time += (time[-1]-time[0])
        if 'deepfake' in  entry['LABEL'].values:
            deepfake_time += (time[-1]-time[0])

    print(f'deepfake time: {deepfake_time/60}')
    print(f'self time: {self_time/60}')

    print("")

def test_data_distribution():
    fig = make_subplots(rows=2, cols=2, shared_yaxes=True)

    dataset_sub1 = Dataset(filepath=PATH, subject='subject1')
    dataset_sub1.preprocess_data(measurement_timeframe ="1000ms")

    dataset_sub2 = Dataset(filepath=PATH, subject='subject2')
    dataset_sub2.preprocess_data(measurement_timeframe ="1000ms")

    dataset_sub3 = Dataset(filepath=PATH, subject='subject3')
    dataset_sub3.preprocess_data(measurement_timeframe ="1000ms")

    dil_sub1 = [len(entry['dilation'].values) for entry in dataset_sub1.data]
    dil_sub2 = [len(entry['dilation'].values) for entry in dataset_sub2.data]
    dil_sub3 = [len(entry['dilation'].values) for entry in dataset_sub3.data]

    dil_all = []
    dil_all.extend(dil_sub1)
    dil_all.extend(dil_sub2)
    dil_all.extend(dil_sub3)

    fig.append_trace (go.Histogram(x=dil_sub1, name='subject1'), 1, 1)
    fig.append_trace(go.Histogram(x=dil_sub2, name= 'subject2'), 1, 2)
    fig.append_trace(go.Histogram(x=dil_sub3, name='subject3'), 2, 1)
    fig.append_trace(go.Histogram(x=dil_all, name='all subjects'), 2, 2)

    # Update x-axes titles
    fig.update_xaxes(title_text="Length l of dilation entries", row=1, col=1)
    fig.update_xaxes(title_text="Length l of dilation entries", row=1, col=2)
    fig.update_xaxes(title_text="Length l of dilation entries", row=2, col=1)
    fig.update_xaxes(title_text="Length l of dilation entries", row=2, col=2)

    # Update y-axes titles
    fig.update_yaxes(title_text="Count of entries per length", row=1, col=1)
    fig.update_yaxes(title_text="Count of entries per length", row=2, col=2)
    fig.update_yaxes(title_text="Count of entries per length", row=2, col=1)
    fig.update_yaxes(title_text="Count of entries per length", row=2, col=2)

    fig.update_layout(
    title=f"Distribution of size of dilation data entries for subject1, subject2, subject3"
)
    fig.show()

def find_best_params_for_svm():
    subjects = ['subject3']
    mss = ['3000ms', '1500ms', '1000ms']

    out = []

    for subject in subjects:
        for ms in mss:
            dataset = Dataset(filepath=PATH, subject=subject)
            dataset.preprocess_data(measurement_timeframe=ms)
            X, y = dataset.feature_extraction()
            classifiers = MultiBaseClassifiers(X=X, y=y)
            params = classifiers.find_best_params()
            params['ms'] = ms
            params['subject'] = subject
            out.append(params)
    print(out)
 
if __name__ == "__main__":
    find_best_params_for_svm()