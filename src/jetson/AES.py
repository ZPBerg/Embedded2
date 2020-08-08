from Crypto.Cipher import AES
from Crypto import Random
import numpy as np
from pbkdf2 import PBKDF2
import salt
import os
from typing import List, Set, Dict, Tuple, Optional


class Encryption():

    def __init__(self):
        '''
        This class handles encryption to prevent identifiable information (facial data)
        from leaving the camera. It also generates a random key that will be used by
        authorized personnel to acces the data.
        '''
        self.salt = os.urandom(16) #Salt variable (Generates a random byte string)
        self.key = PBKDF2("passphrase", self.salt).read(16) #Creates key using KDF scheme

    def encrypt(self,
                coordinates: List[Tuple[int, int, int, int]],
                image: 'numpy.ndarray[numpy.ndarray[numpy.ndarray[numpy.uint8]]]'):
        '''
        This method encrypts the facial regions.
        Args:
            coordinates: bounding box coordinates
            image: Image to be encrypted

        Return:
            image: Encrypted image
            IV: Initialization vector for decrypting image
        '''

        IV = os.urandom(16) #Initialization vector

        mode = AES.MODE_CFB #Sets encryption mode to CFB mode; CFB is great in avoiding the hassle of padding
        encryptor = AES.new(self.key, mode, IV) #Encryptor

        ROI_number = 0
        for c in coordinates:
            x1,y1,x2,y2 = c
            ROI = image[y1:y2, x1:x2]
            img_bytes = np.ascontiguousarray(ROI, dtype=None)

            encData = encryptor.encrypt(img_bytes) #Encrypted data
            data = np.frombuffer(encData, dtype=np.uint8)
            data = np.reshape(data, (y2-y1, x2-x1, 3))
            image[y1:y2, x1:x2] = data

        return image, IV


    def decrypt(self,
                coordinates: List[Tuple[int]],
                image:'numpy.ndarray[numpy.ndarray[numpy.ndarray[numpy.uint8]]]',
                IV: bytes):
        '''
        This method decrypts the facial regions.
        Args:
            coordinates: bounding box coordinates
            image: Image to be decrypted
            IV - Initialization vector

        Return:
            image: Decrypted image
        '''
        mode = AES.MODE_CFB #Sets encryption mode to CFB mode; CFB is great in avoiding the hassle of padding
        decryptor = AES.new(self.key, mode, IV) #Decryptor

        for c in coordinates:
            x1,y1,x2,y2 = c

            ROI = image[y1:y2, x1:x2]
            img_bytes = np.ascontiguousarray(ROI, dtype=None)

            decData = decryptor.decrypt(img_bytes) #Encrypted data
            data = np.frombuffer(decData, dtype=np.uint8)
            data = np.reshape(data, (y2-y1, x2-x1, 3))
            image[y1:y2, x1:x2] = data

        return image
