
from sklearn.metrics import confusion_matrix, make_scorer
import numpy as np


def evaluate_error(preds, gt):
    # Define the cost matrix
    cost_matrix = np.array([[0, 1, 2], 
                            [1, 0, 1], 
                            [2, 1, 0]])
    
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(gt, preds)
    
    # Calculate the error value
    err = np.sum(conf_matrix * cost_matrix) / len(gt)
    
    return err


def get_scorer():
    return make_scorer(evaluate_error,greater_is_better=False)