import cv2
from PIL import Image
import numpy as np
from typing import List, Tuple
import torch
from torchvision import transforms

class Classifier:
    def __init__(self, classifier, cuda: bool):
        """
        Performs classification of facial region into three classes - [Goggles, Glasses, Neither]
        Args:
            classifier - Trained classifier model (Currently, mobilenetv2)
            cuda - True if Nvidia GPU is used
        """
        self.fps = 0
        self.classifier = classifier
        self.device = cuda

    def classifyFace(self,
                     face: np.ndarray):
        """
        This method initializaes the transforms and classifies the face region
        Args:
            face - A 3D numpy array containing facial region

        Return:
            pred - A tensor containing the index of the highest class probability
        """

        classifier = self.classifier

        if 0 in face.shape:
            pass
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(rgb_face)
        # Transforms applied to image before passing it to classifier. These should be
        # the same transforms as applied while training model
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transformed_face = transform(pil_face)
        face_batch = transformed_face.unsqueeze(0)
        device = torch.device("cuda:0" if self.device and torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            face_batch = face_batch.to(device)
            labels = classifier(face_batch)
            m = torch.nn.Softmax(1)
            _, pred = torch.max(labels, 1)

        return pred

    def classifyFrame(self,
                      img: np.ndarray,
                      boxes: List[Tuple[np.float64]]):
        """
        This method loops through all the bounding boxes in an image, calls classifyFace method
        to classify face region and finally draws a box around the face.
        Args:
            img - A 3d numpy array containing input video frame
            boxes - Coordinates of the bounding box around the face

        Return:
            label: Classification label (Goggles, Glasses or Neither)
        """

        label = []
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box[0:4]]
            # draw boxes within the frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            face = img[y1:y2, x1:x2, :]

            label.append(int(self.classifyFace(face).data))

        return label
