import os
import cv2
import numpy as np
import pandas as pd
from heapq import heappush

class OpticDiscDetector:
    def __init__(self, image_path, ground_truth_path, kernel):
        self.paths = sorted(list(map(lambda x: os.path.join(image_path, x), os.listdir(image_path))))
        self.ground_truths = pd.read_csv(ground_truth_path).iloc[:103, 1:3]
        self.kernel = kernel

    def predict(self, index, threshold_start=250, contour_limit=2):
        
        # read image
        image = cv2.imread(self.paths[index])
            
        # convert image into gray and apply historgram equalization
        # since optic disc brighter, its value will be more close to channel limit (255)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)

        # start min threhsold and contour list for iteration
        contour_list = []
        min_threshold = threshold_start

        # continue lowering min threshold and try to find contours
        # until the number of contours reaches the limit
        while len(contour_list) < contour_limit:
            # apply thresholding into equalized image 
            _, thresholded_image = cv2.threshold(equalized_image, min_threshold, 255, cv2.THRESH_BINARY)
            
            # apply erosion to remove structural degradation
            erosion = cv2.erode(thresholded_image, self.kernel, iterations=2)

            # get contours
            contours = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            # calculate contour area to eliminate small ones
            for c in contours:
                area = cv2.contourArea(c)
                if area > 5000:
                    heappush(contour_list, (-area, c))

            # lowering min threshold for next iteration
            min_threshold -= 1

        # apply segmentation to show optic disc
        image[erosion == 255] = [255, 255, 255]

        # find center of fovea using contour and draw a circle
        center = cv2.moments(contour_list[0][1])
        center_x = int(center["m10"] / center["m00"])
        center_y = int(center["m01"] / center["m00"])

        return center_x, center_y, image

class FoveaDetector:
    def __init__(self, image_path, ground_truth_path, kernel):
        self.paths = sorted(list(map(lambda x: os.path.join(image_path, x), os.listdir(image_path))))
        self.ground_truths = pd.read_csv(ground_truth_path).iloc[:103, 1:3]
        self.kernel = kernel

    def predict(self, index, threshold_start=220, contour_limit=2, roi_radius=700):
        
        # read image    
        image = cv2.imread(self.paths[index])
            
        # get the middle point of the image
        mid_x, mid_y = image.shape[1]//2, image.shape[0]//2

        # convert image into negative image to make black points white
        negative_image = 255 - image

        # create circle mask to specify region of interest for fovea
        mask = np.zeros_like(negative_image)
        mask = cv2.circle(mask, (mid_x, mid_y), roi_radius, (255, 255, 255), -1)

        # convert negative image into gray and apply circle mask in the middle area
        gray_image = cv2.cvtColor(negative_image, cv2.COLOR_BGR2GRAY)
        gray_image[mask[:, :, 0] != 255] = 0

        # apply histogram equalization to make brighter area closer to the white
        equalized_image = cv2.equalizeHist(gray_image)

        # start min threhsold and contour list for iteration
        min_threshold = threshold_start
        contour_list = []

        # continue lowering min threshold and try to find contours
        # until the number of contours reaches the limit
        while len(contour_list) < contour_limit:
            # apply thresholding into equalized image 
            _, thresholded_image = cv2.threshold(equalized_image, min_threshold, 255, cv2.THRESH_BINARY)
            
            # apply erosion to remove structural degradation
            erosion = cv2.erode(thresholded_image, self.kernel, iterations=2)

            # get contours 
            contours = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            # calculate contour area to eliminate small ones
            for c in contours:
                area = cv2.contourArea(c)
                if area > 200:
                    heappush(contour_list, (-area, c))
            
            # lowering min threshold for next iteration
            min_threshold =- 1

        # find center of fovea using contour and draw a circle
        center = cv2.moments(contour_list[0][1])
        center_x = int(center["m10"] / center["m00"])
        center_y = int(center["m01"] / center["m00"])

        return center_x, center_y

    def mean_squared_error(self):
        pass

    def f1_score(self, distance_limit=200):
        """
        F1 Score metric to calculate model performance.

        Arguments:
            distance_limit: maximum euclidean distance from ground truth to be accepted as true prediction.
        Returns:
            F1 score between ground truths and predictions 
        """

        # initialize true positive, false positive and false negative (no need true negative)
        tp, fp, fn = 0, 0, 0

        # loop over number of examples
        for i in range(len(self.ground_truths)):
            # get x, y coordinate predictions and ground truths
            x_pred, y_pred = self.predict(i)
            x_true, y_true = self.ground_truths.iloc[i]

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

        return f1