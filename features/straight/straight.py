import numpy as np
import cv2
from features.straight import constants


def kernel_filter(img):
    kernelOpen = np.ones((constants.kernOpenSq, constants.kernOpenSq))
    kernelClose = np.ones((constants.kernClSq, constants.kernClSq))
    maskOpen_dotted = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernelOpen)
    maskClose_dotted = cv2.morphologyEx(maskOpen_dotted, cv2.MORPH_CLOSE, kernelClose)
    return maskClose_dotted


def line_approx(frame):
    if frame is None:
        raise Exception('Error: No frame...')

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 60, 255])

    # Threshold the HSV image to get only yellow colors
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    maskClose_yellow = kernel_filter(mask_yellow)
    maskClose_white = kernel_filter(mask_white)

    final = maskClose_white * (maskClose_yellow == 0) + (maskClose_yellow // 2) * (maskClose_yellow != 0)
    return final
