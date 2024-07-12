import sys
import os
sys.path.append(os.getenv('PATH_TO_TM'))

from dataset.dataset import Dataset
import os
from src.classification.classifiers import MultiBaseClassifiers

def main():
    current_subject = 'subject1'
    measurement_time = ['1500ms']

    PATH =  os.getenv('PATH_TO_TM')+ "/src/data/" 
    SAVE_PATH = os.getenv('PATH_TO_TM')+ "/src/data/results/"
    dataset = Dataset(filepath=PATH, subject=current_subject)

    for ms in measurement_time:
        dataset.preprocess_data(measurement_timeframe=ms)
        X, y = dataset.feature_extraction()
        classifiers = MultiBaseClassifiers(X=X, y=y)
        score_df, prediction_df = classifiers.run()
        score_df.to_csv(SAVE_PATH+current_subject+"/"+current_subject+"_"+ms+".csv", index=False)
        prediction_df.to_csv(SAVE_PATH+current_subject+"/raw_predictions"+"_"+ms+".csv", index=False)

if __name__ == "__main__":
    main()