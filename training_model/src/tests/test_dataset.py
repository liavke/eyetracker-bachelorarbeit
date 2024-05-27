import sys
import os
sys.path.append(os.getenv('PATH_TO_TM'))
from src.dataset.dataset import Dataset
import unittest

PATH = "/Users/liavkeren/bachelor_thesis/bachelor_thesis_code/training_model/src/data/training_client_data"

def test_data_loader():
    pass

def test_preprocessing():
    dataset = Dataset(filepath=PATH)
    dataset.preprocess_data()


class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == "__main__":
    test_preprocessing()