import numpy as np
import pandas as pd

def mean_squared_error(detector, ground_truth_path):
    # read ground truth dataframe
    ground_truth = pd.read_csv(ground_truth_path).iloc[:103, 1:3]

    # get number of examples
    m = len(ground_truth)
    
    # initialize error
    error = 0  

    # loop over number of examples
    for i in range(m):
        # get x, y coordinate predictions and ground truths
        x_pred, y_pred, _ = detector.predict(i)
        x_true, y_true = ground_truth.iloc[i]

        error += (x_true - x_pred)**2 + (y_true - y_pred)**2
    
    return (error / m)

def root_mean_squared_error(detector, ground_truth_path):
    # directly return root of mean squared error
    return np.sqrt(mean_squared_error(detector, ground_truth_path))


def f1_score(detector, ground_truth_path, distance_limit=200):
    """
    F1 Score metric to calculate model performance.

    Arguments:
        distance_limit: maximum euclidean distance from ground truth to be accepted as true prediction.
    Returns:
        F1 score between ground truths and predictions 
    """

    # read ground truth dataframe
    ground_truth = pd.read_csv(ground_truth_path).iloc[:103, 1:3]

    # initialize true positive, false positive and false negative (no need true negative)
    tp, fp, fn = 0, 0, 0

    # loop over number of examples
    for i in range(len(ground_truth)):
        # get x, y coordinate predictions and ground truths
        x_pred, y_pred, _ = detector.predict(i)
        x_true, y_true = ground_truth.iloc[i]

        # calculate euclidean distance
        distance = np.sqrt((x_true - x_pred)**2 + (y_true - y_pred)**2)

        # if distance is less than distance limit, then it is correct (close to correct point)
        # else false and increase both fp and fn because they represent the same thing
        if distance <= distance_limit:
            tp += 1
        else:
            fp += 1
            fn += 1

    # calculate precision and recall, then f1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    return np.round(f1, 4)
