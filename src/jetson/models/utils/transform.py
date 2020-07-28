import cv2
import numpy as np
from typing import Tuple


class BaseTransform:
    def __init__(self, size:int or Tuple[int], mean:Tuple[float]):
        '''
        Transform image to desired size before passing to face detector
        Args:
            size (int) - Desired input size to face detector. If tuple, must be (width, height) NOT (height, width)
            mean (tuple) - mean intensity values of length 3 for 3 channels to perform normalization
        '''

        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, Tuple):
            self.size = size


        if mean is not None:
            self.mean = np.array(mean, dtype=np.float32)
        else:
            self.mean = None

    def __call__(self, image:np.ndarray, boxes=None, labels=None):
        '''
        Returns output of base transform which is the transformed image
        Args:
            image - Image to be transformed
        '''
        return self.base_transform(image), boxes, labels


    def base_transform(self, image:np.ndarray):
        '''
        Calculates base transform by resizing input image and normalizing with mean pixel intensity
        for channels
        Args:
            image - Image to be transformed
        '''
        x = cv2.resize(image, self.size).astype(np.float32)
        if self.mean is not None:
            x -= self.mean
        x = x.astype(np.float32)
        return x
