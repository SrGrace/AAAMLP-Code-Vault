# import pandas and model_selection module of scikit-learn
import pandas as pd
import numpy as np
from sklearn import model_selection


'''
Cross-validation techniques include:
    • k-fold cross-validation
    • stratified k-fold cross-validation
    • hold-out based validation
    • leave-one-out cross-validation
    • group k-fold cross-validation
'''
class CrossValidation(object):
    def __init__(self) -> None:
        return

    '''
    can be used for both classification and regression problems
    -----
    not recommended for imbalanced data-set
    -----
    what if the data-set has large amount of data -> use this as hold-out based validation.
    only change would be to create 10 folds instead of 5 and keep one of those folds as hold-out
    -----
    what if the data-set is too small -> In those cases we can opt for for a type of kfold cross-validation
    where k=N (N: #of samples in the data-set)
    '''
    def kfold(self, df) -> None:
        # df = pd.read_csv("train.csv")

        # we create a new column called kfold and fill it with -1
        df["kfold"] = -1

        # the next step is to randomize the rows of the data
        df = df.sample(frac=1).reset_index(drop=True)

        # initiate the kfold class from model_selection module
        kf = model_selection.KFold(n_splits=5)
        
        # fill the new kfold column
        for fold, (trn_, val_) in enumerate(kf.split(X=df)):
            df.loc[val_, 'kfold'] = fold

        # save the new excel with kfold column 
        df.to_excel("train_folds.xlsx", index=False)
        return

    '''
    choose this blindly for any standard classification problem
    -----
    what if the data-set has large amount of data -> use this as hold-out based validation.
    only change would be to create 10 folds instead of 5 and keep one of those folds as hold-out
    -----
    what if the data-set is too small -> In those cases we can opt for for a type of kfold cross-validation
    where k=N (N: #of samples in the data-set)
    '''
    def stratified_kfold_for_classification(self, df, target_column) -> None:
        # df = pd.read_csv("train.csv")

        # we create a new column called kfold and fill it with -1
        df["kfold"] = -1

        # the next step is to randomize the rows of the data
        df = df.sample(frac=1).reset_index(drop=True)

        # fetch target
        y = df[target_column].values

        # initiate the kfold class from model_selection module
        kf = model_selection.StratifiedKFold(n_splits=5)
        
        # fill the new kfold column
        for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
            df.loc[val_, 'kfold'] = fold

        # save the new excel with kfold column 
        df.to_excel("train_folds.xlsx", index=False)
        return

    '''
    stratified k-fold for a regression problem
    -----
    first divide the target into bins, and then we can use stratified k-fold in the same way as for 
    classification problems.
    '''
    def stratified_kfold_for_regression(self, df, target_column) -> None:
        # we create a new column called kfold and fill it with -1
        df["kfold"] = -1
        
        # the next step is to randomize the rows of the data
        df = df.sample(frac=1).reset_index(drop=True)

        # calculate the number of bins by Sturge's rule
        num_bins = int(np.floor(1 + np.log2(len(df))))

        # bin targets
        df.loc[:, "bins"] = pd.cut(
            df[target_column], bins=num_bins, labels=False
        )
        
        # initiate the kfold class from model_selection module
        kf = model_selection.StratifiedKFold(n_splits=5)
        
        # fill the new kfold column
        # note that, instead of targets, we use bins!
        for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.bins.values)):
            df.loc[val_, 'kfold'] = fold
        
        # drop the bins column
        df = df.drop("bins", axis=1)

        # save the new excel with kfold column 
        df.to_excel("train_folds.xlsx", index=False)
        return 
    
