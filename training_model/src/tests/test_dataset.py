import sys
import os
sys.path.append(os.getenv('PATH_TO_TM'))
from src.dataset.dataset import Dataset
import unittest

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pickle

PATH =  os.getenv('PATH_TO_TM')+ "/src/data/subject1"

def test_data_loader():
    pass

def test_preprocessing():
    dataset = Dataset(filepath=PATH, subject='subject1')
    dataset.preprocess_data()
    return dataset.data

def test_feature_extraction():
    dataset = Dataset(filepath=PATH, subject='subject1')
    dataset.preprocess_data()
    data = dataset.data

    fig = go.Figure()

    for entry in data[0]:
        fig.add_trace(go.Scatter(x=entry['TIME'] ,y = entry['dilation'], mode='lines'))

    fig.show()
    fig.write_html(os.getenv('PATH_TO_TM')+"/visulaisations/29_05_24.html")


class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

def test_visualizing_data():
    PKL_PATH = os.getenv('PATH_TO_TM') + '/src/data/subject1/11_06_dilations.pkl'
    fig = go.Figure()

    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)
        for entry in data:
            if not entry['GAZE_LABEL'].eq('else').any() and len(entry)>400 and entry['GAZE_LABEL'].eq('looking_at_stranger').any():
                fig.add_trace(go.Scatter(x=entry['TIME'] ,y = entry['dilation'], mode='lines'))
    fig.show()

if __name__ == "__main__":
    data = test_preprocessing()
    print("")