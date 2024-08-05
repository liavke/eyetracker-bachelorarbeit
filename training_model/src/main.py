import sys
import os
sys.path.append(os.getenv('PATH_TO_TM'))

from dataset.dataset import Dataset
import os
from src.classification.classifiers import MultiBaseClassifiers, BinaryBaseClassifiers

def main(binary=False):
    current_subject = 'subject3'
    measurement_time = '3000ms'

    PATH =  os.getenv('PATH_TO_TM')+ "/src/data/" 
    SAVE_PATH = os.getenv('PATH_TO_TM')+ "/src/data/results/"
    dataset = Dataset(filepath=PATH, subject=current_subject)

    dataset.preprocess_data(measurement_timeframe=measurement_time)
    X, y = dataset.feature_extraction()
    if binary:
        new_y = [label == 'deepfake' for label in y]
        classifiers = BinaryBaseClassifiers(X=X, y=new_y)
        score_df, prediction_df = classifiers.run()
        score_df.to_csv(SAVE_PATH+current_subject+"/"+current_subject+"_"+measurement_time+".csv", index=False)
        prediction_df.to_csv(SAVE_PATH+current_subject+"/raw_predictions"+"_"+measurement_time+".csv", index=False)

    classifiers = MultiBaseClassifiers(X=X, y=y)
    score_df, prediction_df = classifiers.run()
    score_df.to_csv(SAVE_PATH+current_subject+"/"+current_subject+"_binary_"+measurement_time+".csv", index=False)
    prediction_df.to_csv(SAVE_PATH+current_subject+"/raw_predictions_binary"+"_"+measurement_time+".csv", index=False)
    #classifiers.visualise_roc()


if __name__ == "__main__":
    main(binary=False)