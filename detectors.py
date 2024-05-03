import os
import cv2
import numpy as np
from heapq import heappush

class OpticDiscDetector:
    def __init__(self, image_path):
        self.paths = sorted(list(map(lambda x: os.path.join(image_path, x), os.listdir(image_path))))

    def predict(self, kernel, threshold_start=250, contour_limit=2, output='out'):
        
        for i, path in enumerate(self.paths):
            image = cv2.imread(path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)

            contour_list = []
            min_threshold = threshold_start

            while len(contour_list) < contour_limit:
                _, thresholded_image = cv2.threshold(equalized_image, min_threshold, 255, cv2.THRESH_BINARY)
                erosion = cv2.erode(thresholded_image, kernel, iterations=2)

                contours = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]

                for c in contours:
                    area = cv2.contourArea(c)
                    if area > 5000:
                        heappush(contour_list, (-area, c))
                
                min_threshold -= 1

            image[erosion == 255] = [255, 255, 255]
            center = cv2.moments(contour_list[0][1])
            centerX = int(center["m10"] / center["m00"])
            centerY = int(center["m01"] / center["m00"])
            cv2.circle(image, (centerX, centerY), 9, (0, 0, 255), -1)
            cv2.imwrite(f'{output}{i+1}.jpg', image)

class FoveaDetector:
    def __init__(self, image_path):
        self.paths = sorted(list(map(lambda x: os.path.join(image_path, x), os.listdir(image_path))))

    def predict(self, kernel, threshold_start=220, contour_limit=2, roi_radius=700, output='out'):
        
        for i, path in enumerate(self.paths):
            # read image
            image = cv2.imread(path)
            
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

            # apply erosion to remove structural degradation
            erosion = cv2.erode(equalized_image, kernel, iterations=2)

            # start min threhsold and contour list for iteration
            min_threshold = threshold_start
            contour_list = []

            # continue lowering min threshold and try to find contours
            # until the number of contours reaches the limit
            while len(contour_list) < contour_limit:
                # apply thresholding into equalized image 
                _, thresholded_image = cv2.threshold(erosion, min_threshold, 255, cv2.THRESH_BINARY)

                # get contours 
                contours = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]

                for c in contours:
                    area = cv2.contourArea(c)
                    if area > 200:
                        heappush(contour_list, (-area, c))
                
                # lowering min threshold for next iteration
                min_threshold =- 1

            # find center of fovea using contour and raw a circle
            center = cv2.moments(contour_list[0][1])
            center_x = int(center["m10"] / center["m00"])
            center_y = int(center["m01"] / center["m00"])
            cv2.circle(image, (center_x, center_y), 9, (255, 255, 255), -1)
            
            # save image
            cv2.imwrite(f'{output}{i+1}.jpg', image)

