import pytest
import numpy as np
import cv2
import os
import shutil

from src.jetson.main import writeImg, encryptWorker, fileCount, encryptRet
from src.jetson.encryptor import Encryptor

class TestMain():
    '''
    Tests in this class are for helper functions found in src/main.py
    '''
    def setup(self):
        self.img = np.zeros((300,300,3),np.uint8)
        self.img[:] = (0, 0, 255)
        self.output_dir = "test_output"
        self.encryptor = Encryptor()
        self.boxes = [(10, 10, 20, 20)]
        
    def test_writeImg(self):
        '''
        Tests 'writeImg' function
        Checks:
            - Write directory is created
            - Image is created within write directory
            - Source image and written image are of same shape (written image not null)
            - Filecount is accurate to number of files written
        '''
        fileName = writeImg(self.img, self.output_dir)
        filePath = os.path.join(self.output_dir, fileName)
        assert os.path.isdir(self.output_dir)
        assert os.path.isfile(filePath)
        imgCopy = cv2.imread(filePath)
        assert self.img.shape == imgCopy.shape

        writeImg(self.img, self.output_dir)
        writeImg(self.img, self.output_dir)
        assert fileCount.value == 3

    def test_encryptWorker(self):
        '''
        Tests 'encryptWorker' function
        Checks:
            - Worker increases Filecount
            - Write directory is created
            - Image is created within write directory
            - Source image and written image are of same shape (written image not null)
        '''
        oldFileCount = fileCount.value
        encryptWorker(self.encryptor, self.img, self.boxes, self.output_dir)
        assert fileCount.value - oldFileCount == 1

        image_name, init_vec_list = encryptRet.get()
        filePath = os.path.join(self.output_dir, image_name)
        assert os.path.isdir(self.output_dir)
        assert os.path.isfile(filePath)
        imgCopy = cv2.imread(filePath)
        assert self.img.shape == imgCopy.shape

    def teardown(self):
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        

