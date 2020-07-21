import argparse
import os
import time
import warnings
from multiprocessing import Process, Queue, Value
from threading import Thread
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from src.jetson.AES import Encryption as AESEncryptor
from src.jetson.models.utils.transform import BaseTransform
from src.jetson.models.utils.box_utils import decode, do_nms, postprocess
from src.jetson.models.Retinaface.layers.functions.prior_box import PriorBox
from src.jetson.models.Retinaface.data.config import cfg_mnet as cfg
from src.jetson.models.Retinaface.data.config import cfg_inference as infer_params

from src.db.data_insertion import data_insert

fileCount = Value('i', 0)
# Shared memory queue to allow child encryption process to return to parent
encryptRet = Queue()
DETECTOR_TYPES = ['blazeface', 'retinaface', 'ssd']


class FaceDetector:
    def __init__(self, detector: str, detector_type: str, detection_threshold=0.7, cuda=True, set_default_dev=False):
        """
        Creates a FaceDetector object
        Args:
            detector: A string path to a trained pth file for a ssd model trained in face detection
            detector_type: A DetectorType describing which face detector is being used
            detection_threshold: The minimum threshold for a detection to be considered valid
            cuda: Whether or not to enable CUDA
            set_default_dev: Whether or not to set the default device for PyTorch
        """

        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            if set_default_dev:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device("cpu")
            if set_default_dev:
                torch.set_default_tensor_type('torch.FloatTensor')

        if detector_type == 'ssd':
            from src.jetson.models.SSD.ssd import build_ssd

            self.net = build_ssd('test', 300, 2)
            self.model_name = 'ssd'
            self.net.load_state_dict(torch.load(
                detector, map_location=self.device))
            self.transformer = BaseTransform(self.net.size, (104, 117, 123))

        elif detector_type == 'blazeface':
            from src.jetson.models.BlazeFace.blazeface import BlazeFace

            self.net = BlazeFace(self.device == torch.device("cuda:0"))
            self.net.load_weights(detector)
            self.net.load_anchors("models/BlazeFace/anchors.npy")
            self.model_name = 'blazeface'
            self.net.min_score_thresh = 0.75
            self.net.min_suppression_threshold = 0.3
            self.transformer = BaseTransform(128, None)

        elif detector_type == 'retinaface':
            from src.jetson.models.Retinaface.retinaface import RetinaFace, load_model

            self.net = RetinaFace(cfg=cfg, phase='test')
            self.net = load_model(
                self.net, detector, load_to_cpu=self.device == torch.device("cpu"))
            self.model_name = 'retinaface'
            self.image_shape = infer_params["image_shape"]  # (H, W)
            self.resize = infer_params["resize"]
            self.transformer = BaseTransform(
                (self.image_shape[1], self.image_shape[0]), (104, 117, 123))
            priorbox = PriorBox(cfg, image_size=self.image_shape)
            priors = priorbox.forward()
            self.prior_data = priors.data.to(self.device)

        self.detection_threshold = detection_threshold
        self.net.to(self.device)
        self.net.eval()

    def detect(self,
               image: np.ndarray):
        """
        Performs face detection on the image passed
        Args:
            image: A 3D numpy array representing an image

        Return:
            The bounding boxes of the face(s) that were detected formatted (upper left corner(x, y) , lower right corner(x,y))
        """

        if self.model_name == 'ssd':
            x = torch.from_numpy(self.transformer(image)[0]).permute(2, 0, 1)
            x = Variable(x.unsqueeze(0)).to(self.device)
            y = self.net(x)
            detections = y.data
            scale = torch.Tensor(
                [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            bboxes = []
            j = 0
            while j < detections.shape[2] and detections[0, 1, j, 0] > self.detection_threshold:
                pt = (detections[0, 1, j, 1:] * scale).cpu().numpy()
                x1, y1, x2, y2 = pt
                conf = detections[0, 1, j, 0].item()
                bboxes.append((x1, y1, x2, y2, conf))
                j += 1

            return bboxes

        elif self.model_name == 'blazeface':
            img = self.transformer(image)[0].astype(np.float32)

            detections = self.net.predict_on_image(img)
            if isinstance(detections, torch.Tensor):
                detections = detections.cpu().numpy()

            if detections.ndim == 1:
                detections = np.expand_dims(detections, axis=0)

            bboxes = []
            for i in range(detections.shape[0]):
                ymin = detections[i, 0] * image.shape[0]
                xmin = detections[i, 1] * image.shape[1]
                ymax = detections[i, 2] * image.shape[0]
                xmax = detections[i, 3] * image.shape[1]
                conf = detections[i, 16]

                img = img / 127.5 - 1.0

                for k in range(6):
                    kp_x = detections[i, 4 + k * 2] * img.shape[1]
                    kp_y = detections[i, 4 + k * 2 + 1] * img.shape[0]

                bboxes.append((xmin, ymin, xmax, ymax, conf))

            return bboxes

        elif self.model_name == 'retinaface':
            img = (self.transformer(image)[0]).transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            loc, conf, _ = self.net(
                img)  # forward pass: Returns bounding box location, confidence and facial landmark locations

            boxes = decode(loc.data.squeeze(
                0), self.prior_data, cfg['variance'])
            boxes, scores = postprocess(
                boxes, conf, self.image_shape, self.detection_threshold, self.resize)
            dets = do_nms(boxes, scores, infer_params["nms_thresh"])

            bboxes = [tuple(det[0:5]) for det in dets]

            return bboxes


class VideoCapturer(object):
    def __init__(self, src=0):
        """
        This class captures videos using open-cv's VideoCapture object
        Args:
            src: This is the connection to the source of the video stream (webcam or raspberry pi camera)
        """

        self.capture = cv2.VideoCapture(src)
        _, self.frame = self.capture.read()
        self.running = Value('b', True)
        self.t1 = Thread(target=self.update, args=())
        self.t1.daemon = True
        self.t1.start()

    def update(self):
        """Get next frame in video stream"""
        while self.running.value:
            if self.capture.isOpened():
                _, self.frame = self.capture.read()
            time.sleep(.01)
        print("WOW")

    def get_frame(self):
        """ Return current frame in video stream"""
        return self.frame

    def close(self):
        self.running.value = False
        self.t1.join()


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
            transforms.RandomGrayscale(1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        transformed_face = transform(pil_face)
        face_batch = transformed_face.unsqueeze(0)
        device = torch.device(
            "cuda:0" if self.device and torch.cuda.is_available() else "cpu")
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

            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            face = img[y1:y2, x1:x2, :]

            label.append(int(self.classifyFace(face).data))

        return label


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
        """

        encryptedImg, _ = self.encryptor.encrypt(coordinates, img)

        return encryptedImg

    def encryptFrame(self, img: np.ndarray,
                     boxes: List[Tuple[np.float64]]):
        """
        This method takes the face coordinates, encrypts the facial region, writes encrypted image to file filesystem
        Args:
            img: A 3D numpy array containing image to be encrypted
            boxes: facial Coordinates
        """
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box[0:4]]
            # draw boxes within the frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            img = self.encryptFace([(x1, y1, x2, y2)], img)

        return img


def writeImg(img, output_dir):
    """
    This method is used to write an image to an output directory
    Args:
        img: A 3D numpy array containing image to be written
        output_dir: directory to be written to
    Ret:
        face_file_name: os path to written file
    """
    if not os.path.isdir(output_dir):
        os.mkdir(args.output_dir)
    global fileCount
    face_file_name = os.path.join(output_dir, f'{fileCount.value}.jpg')

    # TODO: Remove this print statement after db integration
    print("writing ", face_file_name)
    if args.write_imgs:
        cv2.imwrite(face_file_name, img)
    with fileCount.get_lock():
        fileCount.value += 1

    return face_file_name


def encryptWorker(encryptor, img, boxes, output_dir, write_imgs):
    """
    This method is intended to be spawned as a separate process to handle encrypting and writing of individual frames
    Args:
        encryptor: an encryptor object that contains an AES encryptor object and decryption key
        img: A 3D numpy array containing an image to be enrypted and written
        boxes: facial Coordinates
        output_dir: directory to be written to
    """
    encryptedImg = encryptor.encryptFrame(img, boxes)
    if write_imgs:
        writtenImg = writeImg(encryptedImg, output_dir)
        encryptRet.put(writtenImg)


def drawFrame(boxes, frame, fps):
    """
    This method is used to draw the video detection frame viewable by the user
    Args:
        boxes: facial Coordinates
        frame: current frame from video capturer being processed
        fps: frames per second the detector is capable of detecting, classifying, and encrypting
    """
    class_names = ['Glasses', 'Goggles', 'Neither']
    index = 0
    for box in boxes:
        frame = cv2.putText(frame,
                            'label: %s' % class_names[label[index]],
                            (int(box[0]), int(box[1] - 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255))

        frame = cv2.putText(frame,
                            'fps: %.3f' % fps,
                            (int(box[0]), int(box[1] - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255))

        index += 1

    cv2.imshow("Face Detect", frame)


if __name__ == "__main__":
    warnings.filterwarnings("once")
    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument('--detector', '-d', type=str, required=True,
                        help="Path to a trained face detector .pth file")
    parser.add_argument('--detector_type', '-t', type=str, required=True, help="Type of face detector. One of "
                                                                               "blazeface, ssd, or retinaface.")
    parser.add_argument('--cuda', '-c', default=False,
                        action='store_true', help="Enable cuda")
    parser.add_argument('--classifier', type=str,
                        help="Path to a trained classifier .pth file")
    parser.add_argument('--write_imgs', default=False,
                        help='Write images to output_dir')
    parser.add_argument('--output_dir', default='encrypted_imgs',
                        type=str, help="Where to output encrypted images")
    args = parser.parse_args()

    if args.detector_type not in DETECTOR_TYPES:
        print(
            'Please include a valid detector type (\'blazeface\', \'ssd\', or \'retinaface\'')
        exit(1)

    device = torch.device('cpu')
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')

    g = torch.load(args.classifier, map_location=device)
    g.eval()

    capturer = VideoCapturer()
    detector = FaceDetector(detector=args.detector, detector_type=args.detector_type,
                            cuda=args.cuda and torch.cuda.is_available(), set_default_dev=True)
    classifier = Classifier(g, args.cuda)
    encryptor = Encryptor()

    run_face_detection: bool = True
    while run_face_detection:  # main video detection loop that will iterate until ESC key is entered
        start_time = time.time()

        frame = capturer.get_frame()
        boxes = detector.detect(frame)

        # copy memory for encrypting image separate from unencrypted image
        encryptedImg = frame.copy()

        if len(boxes) != 0:
            p1 = Process(target=encryptWorker, args=(
                encryptor, encryptedImg, boxes, args.output_dir, args.write_imgs))
            p1.daemon = True
            p1.start()

            label = classifier.classifyFrame(frame, boxes)

            fps = 1 / (time.time() - start_time)
            drawFrame(boxes, frame, fps)

            # remove frame creation and drawing before deployment

            p1.join()
            if cv2.waitKey(1) == 27:
                run_face_detection = False

    capturer.close()
    cv2.destroyAllWindows()
    exit(0)
