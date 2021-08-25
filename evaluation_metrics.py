import logging
import numpy as np
from sklearn import metrics


'''
If we talk about classification problems, the most common metrics used are:
    - Accuracy
    - Precision (P)
    - Recall (R)
    - F1 score (F1)
    - Area under the ROC (Receiver Operating Characteristic) curve or simply AUC (AUC)
    - Log loss - Precision at k (P@k)
    - Average precision at k (AP@k)
    - Mean average precision at k (MAP@k)

When it comes to regression, the most commonly used evaluation metrics are:
    - Mean absolute error (MAE)
    - Mean squared error (MSE)
    - Root mean squared error (RMSE)
    - Root mean squared logarithmic error (RMSLE)
    - Mean percentage error (MPE)
    - Mean absolute percentage error (MAPE)
    - R2
'''
class EvaluationMetrics(object):
    def __init__(self) -> None:
        return

    '''
    CLASSIFICATION METRICS
    -----
    When we have an equal number of positive and negative samples in a binary classification metric, 
    we generally use accuracy, precision, recall and f1.
    '''
    def accuracy(self, y_true, y_pred, module_flag="sklearn") -> float:
        """
            Function to calculate accuracy
            :param y_true: list of true values
            :param y_pred: list of predicted values
            :param module_flag: custom or sklearn
            :return: accuracy score
        """
        if module_flag == "sklearn":
            return metrics.accuracy_score(y_true, y_pred)
        
        else:
            # initialize a simple counter for correct predictions
            correct_counter = 0
            # loop over all elements of y_true and y_pred "together"
            for yt, yp in zip(y_true, y_pred):
                if yt == yp:
                # if prediction is equal to truth, increase the counter
                    correct_counter += 1

            # return accuracy which is correct predictions over the number of samples
            return correct_counter / len(y_true)

    '''
    these function for binary classification only
    '''
    def true_positive(self, y_true, y_pred):
        """
            Function to calculate True Positives
            :param y_true: list of true values
            :param y_pred: list of predicted values
            :return: number of true positives
        """
        # initialize
        tp = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1:
                tp += 1
        return tp

    def true_negative(self, y_true, y_pred):
        """
            Function to calculate True Negatives
            :param y_true: list of true values
            :param y_pred: list of predicted values
            :return: number of true negatives
        """
        # initialize
        tn = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 0 and yp == 0:
                tn += 1
        return tn
    
    def false_positive(self, y_true, y_pred):
        """
            Function to calculate False Positives
            :param y_true: list of true values
            :param y_pred: list of predicted values
            :return: number of false positives
        """
        # initialize
        fp = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 0 and yp == 1:
                fp += 1
        return fp

    def false_negative(self, y_true, y_pred):
        """
            Function to calculate False Negatives
            :param y_true: list of true values
            :param y_pred: list of predicted values
            :return: number of false negatives
        """
        # initialize
        fn = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 0:
                fn += 1
        return fn
    
    def accuracy_v2(self, y_true, y_pred):
        """
            Function to calculate accuracy using tp/tn/fp/fn
            :param y_true: list of true values
            :param y_pred: list of predicted values
            :return: accuracy score
        """
        tp = self.true_positive(y_true, y_pred)
        fp = self.false_positive(y_true, y_pred)
        fn = self.false_negative(y_true, y_pred)
        tn = self.true_negative(y_true, y_pred)
        accuracy_score = (tp + tn) / (tp + tn + fp + fn)
        return accuracy_score
    
    def precision(self, y_true, y_pred, module_flag="sklearn"):
        """
            Function to calculate precision
            :param y_true: list of true values
            :param y_pred: list of predicted values
            :return: precision score
        """
        if module_flag == 'sklearn':
            return metrics.precision_score(y_true, y_pred)
        else:
            tp = self.true_positive(y_true, y_pred)
            fp = self.false_positive(y_true, y_pred)
            precision = tp / (tp + fp)
            return precision

    def recall(self, y_true, y_pred, module_flag="sklearn"):
        """
            Function to calculate recall
            :param y_true: list of true values
            :param y_pred: list of predicted values
            :return: recall score
        """
        if module_flag == 'sklearn':
            return metrics.recall_score(y_true, y_pred)
        else:
            tp = self.true_positive(y_true, y_pred)
            fn = self.false_negative(y_true, y_pred)
            recall = tp / (tp + fn)
            return recall
    
    '''
    metrics to use when skewed data-set
    '''
    def f1(self, y_true, y_pred, module_flag="sklearn"):
        """
            Function to calculate f1 score
            :param y_true: list of true values
            :param y_pred: list of predicted values
            :return: f1 score
        """
        if module_flag == 'sklearn':
            return metrics.f1_score(y_true, y_pred)
        else:
            p = self.precision(y_true, y_pred)
            r = self.recall(y_true, y_pred)
            score = 2 * p * r / (p + r)
            return score
    
    def auc(self, y_true, y_pred, module_flag="sklearn"):
        """
            Function to calculate roc_auc_score
            :param y_true: list of true values
            :param y_pred: list of predicted values
            :return: roc_auc_score
        """
        if module_flag == 'sklearn':
            return metrics.auc(y_true, y_pred)
        else:
            return

    def roc_auc_score(self, y_true, y_pred, module_flag="sklearn"):
        """
            Function to calculate roc_auc_score
            :param y_true: list of true values
            :param y_pred: list of predicted values
            :return: roc_auc_score
        """
        if module_flag == 'sklearn':
            return metrics.roc_auc_score(y_true, y_pred)
        else:
            return
    
    def confusion_matrix(self, y_true, y_pred, module_flag="sklearn"):
        """
            Function to calculate confusion_matrix
            :param y_true: list of true values
            :param y_pred: list of predicted values
            :return: confusion_matrix
        """
        if module_flag == 'sklearn':
            return metrics.confusion_matrix(y_true, y_pred)
        else:
            return

    '''
    for multi-label classification
        - precision at k (p@k)
        - average precision at k (ap@k)
        - mean average precision at k (map@k)
        - log loss
    '''
    def log_loss(self, y_true, y_proba, module_flag="sklearn"):
        """
            Function to calculate log loss
            :param y_true: list of true values
            :param y_proba: list of probabilities for 1
            :return: overall log loss
        """
        if module_flag == 'sklearn':
            return metrics.log_loss(y_true, y_proba)
        else:
            # define an epsilon value, this can also be an input, this value is used to clip probabilities
            epsilon = 1e-15

            # initialize empty list to store individual losses
            loss = []

            # loop over all true and predicted probability values
            for yt, yp in zip(y_true, y_proba):
                # adjust probability: 0 gets converted to 1e-15, 1 gets converted to 1-1e-15
                yp = np.clip(yp, epsilon, 1 - epsilon)
                # calculate loss for one sample
                temp_loss = - 1.0 * (
                    yt * np.log(yp) + (1 - yt) * np.log(1 - yp)
                )
                # add to loss list
                loss.append(temp_loss)

            # return mean loss over all samples
            return np.mean(loss)
    
    def pk(self, y_true, y_pred, k):
        """
            This function calculates precision at k for a single sample
            :param y_true: list of values, actual classes
            :param y_pred: list of values, predicted classes
            :param k: the value for k
            :return: precision at a given value k
        """
        # if k is 0, return 0. we should never have this as k is always >= 1
        if k == 0:
            return 0
        # we are interested only in top-k predictions
        y_pred = y_pred[:k]
        # convert predictions to set
        pred_set = set(y_pred)
        # convert actual values to set
        true_set = set(y_true)
        # find common values
        common_values = pred_set.intersection(true_set)
        # return length of common values over k
        return len(common_values) / len(y_pred[:k])
    
    def apk(self, y_true, y_pred, k):
        """
            This function calculates average precision at k 
            for a single sample
            :param y_true: list of values, actual classes
            :param y_pred: list of values, predicted classes
            :return: average precision at a given value k
        """
        # initialize p@k list of values
        pk_values = []
        # loop over all k. from 1 to k + 1
        for i in range(1, k + 1):
            # calculate p@i and append to list
            pk_values.append(self.pk(y_true, y_pred, i))
        # if we have no values in the list, return 0
        if len(pk_values) == 0:
            return 0
        # else, we return the sum of list over length of list
        return sum(pk_values) / len(pk_values)
    
    def mapk(self, y_true, y_pred, k):
        """
            This function calculates mean avg precision at k 
            for a single sample
            :param y_true: list of values, actual classes
            :param y_pred: list of values, predicted classes
            :return: mean avg precision at a given value k
        """
        # initialize empty list for apk values
        apk_values = []
        # loop over all samples
        for i in range(len(y_true)):
        # store apk values for every sample
            apk_values.append(
                self.apk(y_true[i], y_pred[i], k=k)
            )
        # return mean of apk values list
        return sum(apk_values) / len(apk_values)
    
    '''
    REGRESSION METRICS
    -----
    '''
    def mean_absolute_error(self, y_true, y_pred, module_flag="sklearn"):
        """
            This function calculates mae
            :param y_true: list of real numbers, true values
            :param y_pred: list of real numbers, predicted values
            :return: mean absolute error
        """
        if module_flag == 'sklearn':
            return metrics.mean_absolute_error(y_true, y_pred)
        
        else:
            # initialize error at 0
            error = 0
            # loop over all samples in the true and predicted list
            for yt, yp in zip(y_true, y_pred):
                # calculate absolute error and add to error
                error += np.abs(yt - yp)

            # return mean error
            return error / len(y_true)
    
    def mean_squared_error(self, y_true, y_pred, module_flag="sklearn"):
        """
            This function calculates mse
            :param y_true: list of real numbers, true values
            :param y_pred: list of real numbers, predicted values
            :return: mean squared error
        """
        if module_flag == 'sklearn':
            return metrics.mean_squared_error(y_true, y_pred)
        
        else:
            # initialize error at 0
            error = 0
            # loop over all samples in the true and predicted list
            for yt, yp in zip(y_true, y_pred):
                # calculate squared error and add to error
                error += (yt - yp) ** 2

            # return mean error
            return error / len(y_true)
    
    def mean_squared_log_error(self, y_true, y_pred, module_flag="sklearn"):
        """
        This function calculates msle
        :param y_true: list of real numbers, true values
        :param y_pred: list of real numbers, predicted values
        :return: mean squared logarithmic error
        """
        if module_flag == 'sklearn':
            return metrics.mean_squared_log_error(y_true, y_pred)
        
        else:
            # initialize error at 0
            error = 0
            # loop over all samples in true and predicted list
            for yt, yp in zip(y_true, y_pred):
                # calculate squared log error and add to error
                error += (np.log(1 + yt) - np.log(1 + yp)) ** 2

            # return mean error
            return error / len(y_true)
    
    def mean_abs_percentage_error(self, y_true, y_pred, module_flag="sklearn"):
        """
            This function calculates mpe
            :param y_true: list of real numbers, true values
            :param y_pred: list of real numbers, predicted values
            :return: mean abs percentage error
        """
        if module_flag == 'sklearn':
            return metrics.mean_absolute_percentage_error(y_true, y_pred)
        
        else:
            # initialize error at 0
            error = 0
            # loop over all samples in true and predicted list
            for yt, yp in zip(y_true, y_pred):
                # calculate percentage error and add to error
                error += np.abs(yt - yp) / yt

            # return mean abs percentage error
            return error / len(y_true)
    
    def r2(self, y_true, y_pred, module_flag="sklearn"):
        """
        This function calculates r-squared score
        :param y_true: list of real numbers, true values
        :param y_pred: list of real numbers, predicted values
        :return: r2 score
        """
        if module_flag == 'sklearn':
            return metrics.r2_score(y_true, y_pred)
        
        else:
            # calculate the mean value of true values
            mean_true_value = np.mean(y_true)
            # initialize numerator & denominator with 0
            numerator, denominator = 0, 0
            
            # loop over all true and predicted values
            for yt, yp in zip(y_true, y_pred):
                # update numerator
                numerator += (yt - yp) ** 2
                # update denominator
                denominator += (yt - mean_true_value) ** 2

            # return 1 - ratio
            return 1 - (numerator / denominator)
    
    '''
    some advance metrics: can be used for both classification and regression
    '''
    def cohen_kappa(self, y_true, y_pred, module_flag="sklearn"):
        """
        This function calculates cohen_kappa score
        :param y_true: list of real numbers, true values
        :param y_pred: list of real numbers, predicted values
        :return: cohen_kappa score
        """
        if module_flag == 'sklearn':
            return metrics.cohen_kappa_score(y_true, y_pred)
        
        else:
            return
    
    def mcc(self, y_true, y_pred, module_flag="sklearn"):
        """
            This function calculates Matthew's Correlation Coefficient
            for binary classification.
            :param y_true: list of true values
            :param y_pred: list of predicted values
            :return: mcc score
        """
        if module_flag == 'sklearn':
            return metrics.matthews_corrcoef(y_true, y_pred)
        
        else:
            tp = self.true_positive(y_true, y_pred)
            tn = self.true_negative(y_true, y_pred)
            fp = self.false_positive(y_true, y_pred)
            fn = self.false_negative(y_true, y_pred)
            numerator = (tp * tn) - (fp * fn)
            denominator = (
                (tp + fp) * (fn + tn) * (fp + tn) * (tp + fn)
            )
            denominator = denominator ** 0.5
            return numerator/denominator




