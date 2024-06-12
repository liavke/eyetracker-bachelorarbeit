from classification.classifiers import SVM_Classifier, K_Means_Classifier, Naive_Bayes_Classifier, Decision_Trees_Classifier
from dataset.dataset import Dataset
import os

def main():
    subject = 'subject1'
    PATH =  os.getenv('PATH_TO_TM')+ "/src/data/" + subject
    dataset = Dataset(filepath=PATH)
    dataset.preprocess_data()
    X, y = dataset.feature_extraction()

if __name__ == "__main__":
    main()