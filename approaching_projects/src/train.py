# train.py
import os
import sys
import joblib
import argparse
import pandas as pd
from datetime import datetime

import config
import model_dispatcher

sys.path.append('../../')
from evaluation_metrics import EvaluationMetrics


def run(fold, model):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # training data is where kfold is not equal to provided fold also, note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column from dataframe and convert it to a numpy array by using .values.
    # target is label column in the dataframe
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values

    # similarly, for validation, we have
    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values

    # initialize simple decision tree classifier from sklearn
    # clf = tree.DecisionTreeClassifier()

    # fetch the model from model_dispatcher
    clf = model_dispatcher.models[model]

    # for the model on training data
    clf.fit(x_train, y_train)

    # create predictions for validation samples
    preds = clf.predict(x_valid)

    # calculate & print accuracy | classification report
    # accuracy = EvaluationMetrics.accuracy(y_valid, preds)
    classification_report = EvaluationMetrics.classification_report(y_valid, preds)
    print("Fold={}\n, Accuracy={}\n".format(fold, classification_report))

    # save the model
    joblib.dump(
        clf, 
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )

if __name__ == "__main__":
    start = datetime.now()

    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser(description='Training Module')

    # add the different arguments you need and their type
    parser.add_argument("-f", "--fold", type=int, help='kfold fold field', required=True)
    parser.add_argument("-m", "--model", type=int, help='model field', required=True)


    # read the arguments from the command line
    args = parser.parse_args()

    # run the fold specified by command line arguments
    run(
        fold=args.fold,
        model=args.model
    )

    print("Process completed in {} seconds\n".format(str((datetime.now() - start))))


