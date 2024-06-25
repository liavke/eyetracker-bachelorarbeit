import sys
import os
sys.path.append(os.getenv('PATH_TO_TM'))

from dataset.dataset import Dataset
import os
from src.classification.classifiers import MultiBaseClassifiers

def main():
    current_subject = 'subject2'
    PATH =  os.getenv('PATH_TO_TM')+ "/src/data/" + current_subject
    SAVE_PATH = os.getenv('PATH_TO_TM')+ "/src/data/"+current_subject+".csv"
    dataset = Dataset(filepath=PATH, subject=current_subject)
    dataset.preprocess_data()
    X, y = dataset.feature_extraction()
    classifiers = MultiBaseClassifiers(X=X, y=y)
    score_df = classifiers.run()
    score_df.to_csv(SAVE_PATH, index=False)

if __name__ == "__main__":
    main()