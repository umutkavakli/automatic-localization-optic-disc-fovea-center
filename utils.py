import cv2
import numpy as np
import pandas as pd

def mean_squared_error(detector, ground_truth_path):
    """
    Calculates mean squared error (MSE).

    Arguments:
        detector: One of the object of OpticDiscDetector or Fovea Detector.
        ground_truth_path: Relative path of corresponding ground truths stored in .csv file.
    Returns:
        Average of squared error for all samples.
    """

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
    """
    Calculates root mean squared error (RMSE) by getting MSE first, then taking the square root of it.

    Arguments:
        detector: One of the object of OpticDiscDetector or Fovea Detector.
        ground_truth_path: Relative path of corresponding ground truths stored in .csv file.
    Returns:
        Square root of MSE for all samples.
    """

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

def save_image(detector, index, radius=11, output='out', return_mask=False):
    """
    Saving the prediction of one image corresponding to its index value.

    Arguments:
        detector: One of the object of OpticDiscDetector or Fovea Detector.
        index: One of the index value in total images.
        radius: Radius size for the prediction of center x, y location.
        output: Output path.
        return_mask: When it is true, it returns segmented area of image. This can be used only for OpticDiscDetector.
    Returns:
        None. It saves an image with the marked x, y location by drawing a circle. Also, it saves segmented optic disc if return_mask=True. 
    """
    
    # return mask is available only for optic disc detection
    # then if it is true, detector should be optic detector
    if return_mask:
        x, y, image = detector.predict(index, return_mask=return_mask)
    else:
        x, y, image = detector.predict(index)

    # mark the center of area with circle
    cv2.circle(image, (x, y), radius, (255, 0, 0), -1)
    
    # same image
    cv2.imwrite(f'{output}{index+1}.jpg', image)

def save_all_images(detector, radius=11, output='out', return_mask=False):
    """
    Saving all images using save_image method.

    Arguments:
        detector: One of the object of OpticDiscDetector or Fovea Detector.
        radius: Radius size for the prediction of center x, y location.
        output: Output path.
        return_mask: When it is true, it returns segmented area of image. This can be used only for OpticDiscDetector.
    Returns:
        None. It saves all images with the marked x, y location by drawing a circle. Also, it saves segmented optic disc if return_mask=True. 
    """

    for i in range(len(detector)):
        save_image(detector, i, radius, output, return_mask)