from __future__ import print_function

import argparse
import csv
import os

import cv2
import torch
from tqdm import tqdm

from src.jetson.face_detector import FaceDetector
from scripts.utils import check_rotation, correct_rotation

"""
Run the face detector model on a folder of videos (most recently used on TestVideos from the Drive).
Save bbox detections to a csv file to be compared in evaluation.py
An earlier version of this script was used to compare Retinaface with
a Mobilenet backbone versus a Resnet backbone; comparison of object
detectors would be its most applicable use.
"""

DETECTIONS_FILE = 'detection_results.csv'


def create_directory(root_directory):
    if not os.path.isdir(root_directory):
        os.mkdir(root_directory)


def get_videos(input_directory):
    filenames = []
    for dirName, subdirList, fileList in os.walk(input_directory):
        for filename in fileList:
            ext = '.' + filename.split('.')[-1]
            if ext in ['.mov', '.mp4', '.avi', '.MOV', '.MP4', '.AVI']:
                filenames.append(dirName + '/' + filename)

    return filenames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save face detection results')
    parser.add_argument('--detector', '-d', type=str, required=True, help="Path to a trained face detector .pth file")
    parser.add_argument('--detector_type', '-t', type=str, required=True, help="Type of face detector. One of "
                                                                               "blazeface, ssd, or retinaface.")
    parser.add_argument('--cuda', '-c', action="store_true", default=False, help='Use CUDA')
    parser.add_argument('--input_directory', '-i', default='test_videos/', type=str,
                        help='directory where test videos are located')
    parser.add_argument('--output_directory', '-o', default='ground_truth_detections_lowlight/', type=str,
                        help='directory to store detected labels')

    args = parser.parse_args()

    device = torch.device('cuda:0') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    create_directory(args.output_directory)

    torch.set_grad_enabled(False)

    detector = FaceDetector(detector=args.detector, detector_type=args.detector_type,
                            cuda=args.cuda and torch.cuda.is_available(), set_default_dev=True)

    video_files = get_videos(args.input_directory)

    for video in video_files:
        print("Video: ", video)
        cap = cv2.VideoCapture(video)
        rotate_code = check_rotation(video)
        file_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        detections = []

        for frame_num in tqdm(range(file_len)):
            _, frame = cap.read()
            if rotate_code is not None:
                frame = correct_rotation(frame, rotate_code)
            boxes = detector.detect(frame)
            detection = [video, frame_num]

            # each box is one set of face coords
            for box in boxes:
                for b in box:
                    detection.append(b)
            detections.append(detection)

        # save detections to csv one video at a time
        with open(DETECTIONS_FILE, "a") as f:
            writer = csv.writer(f)
            writer.writerows(detections)

    exit(0)
