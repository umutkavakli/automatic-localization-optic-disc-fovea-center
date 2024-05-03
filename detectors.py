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

