#take in np array image, and give the relative smoothness
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def alpha_filter_low(image, alpha):
    # turn any pixel below alpha to 0
    image[image < alpha] = 0
    return image

def alpha_filter_high(image, alpha):
    # turn any pixel above alpha to 0
    image[image > alpha] = 0
    return image

def get_smoothness(image):
    #apply gaussian blur
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    #calculate laplacian
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    #calculate variance
    variance = laplacian.var()
    return variance