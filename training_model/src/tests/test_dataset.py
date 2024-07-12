import sys
import os
sys.path.append(os.getenv('PATH_TO_TM'))
from src.dataset.dataset import Dataset
import unittest

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from collections import Counter

PATH =  os.getenv('PATH_TO_TM')+ "/src/data/"

def test_data_loader():
    pass

def test_preprocessing():
    dataset = Dataset(filepath=PATH, subject='subject1')
    dataset.preprocess_data()
    return dataset.data

def test_feature_extraction():
    dataset = Dataset(filepath=PATH, subject='subject3')
    dataset.preprocess_data(measurement_timeframe="1000ms")
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
    dataset = Dataset(filepath=PATH, subject='subject2')
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

if __name__ == "__main__":
    test_visualizing_data()
    print("")