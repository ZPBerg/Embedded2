from src.jetson.AES import Encryption as AESEncryptor
import numpy as np
from typing import List, Tuple

class Encryptor(object):
    def __init__(self):
        """
        This class acts as a wrapper for the AES encryptor in AES.py and stores the encryption key for decrypting
        """
        self.encryptor = AESEncryptor()
        self.key = self.encryptor.key

    def encryptFace(self, coordinates: List[Tuple[int]],
                    img: np.ndarray):
        """
        This function Encrypts faces
        Args:
            coordinates - Face coordinates returned by face detector
            img - A 3D numpy array containing image to be encrypted

        Return:
            encryptedImg - Image with face coordinates encrypted
            init_vec - Initialization vector generated for face
        """

        encryptedImg, init_vec = self.encryptor.encrypt(coordinates, img)

        return encryptedImg, init_vec

    def encryptFrame(self, img: np.ndarray,
                     boxes: List[Tuple[np.float64]]):
        """
        This method takes the face coordinates, encrypts the facial region, writes encrypted image to file filesystem
        Args:
            img: A 3D numpy array containing image to be encrypted
            boxes: facial Coordinates

        Return:
            img - Original image with all faces encrypted
            init_vec_list - list of initialization vectors for each face in image
        """
        init_vec_list = []
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box[0:4]]
            # draw boxes within the frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            img, init_vec = self.encryptFace([(x1, y1, x2, y2)], img)
            init_vec_list.append(init_vec)

        return img, init_vec_list
