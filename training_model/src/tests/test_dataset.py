import sys
import os
sys.path.append(os.getenv('PATH_TO_TM'))
from src.dataset.dataset import Dataset
import unittest

import plotly.graph_objects as go
import plotly.express as px
import numpy as np



PATH =  os.getenv('PATH_TO_TM')+ "/src/data/training_client_data/today"

def test_data_loader():
    pass

def test_preprocessing():
    dataset = Dataset(filepath=PATH)
    dataset.preprocess_data()
    return dataset

def test_feature_extraction():
    dataset = Dataset(filepath=PATH)
    dataset.preprocess_data()

    x,y = dataset.test_feature_extraction()
    fig = go.Figure()

    scatter_x = np.arange(max(len(df) for df in x[0]))

    for entry in x[0]:
        fig.add_trace(go.Scatter(x=scatter_x ,y = entry['dilation'], mode='lines'))

    fig.show()

    x = x[0]
    fig = px.box(y = [dilation['dilation'] for dilation in x], 
                 x=[time['TIME'] for time in x])
    fig.show()

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == "__main__":
    test_feature_extraction()