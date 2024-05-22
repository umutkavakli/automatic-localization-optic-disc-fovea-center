import cv2

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
    cv2.circle(image, (x, y), radius, (255, 255, 255), -1)
    
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