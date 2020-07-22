import argparse
import json
import os
import warnings

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from scripts.goggle_classifier import get_model
from scripts.utils import check_rotation, correct_rotation
from src.jetson.main import FaceDetector, Classifier

DETECTIONS_FILE = 'det_results.txt'
CLASSIFICATION_RESULTS_FILE = 'results.json'
VIDEO_EXT = ['.mov', '.mp4', '.avi', '.MOV', '.MP4', '.AVI']

"""
Use this script with annotator.py. 
Videos to be evaluated should be from the TestVideos folder on the Drive.
"""

# TODO - TODO TODO don't evaluate face detection? Would have to manually label faces but we're using a
# TODO - SOTA face detection model that could just empirically be observed to work
# TODO make comments with @param things
# TODO load detections as csv file


class Evaluator():
    def __init__(self, cuda, detector, detector_type, classifier, input_directory, annotation_path, rate=1):
        """
        Evaluates face detection and goggle classification performance.
        Goggle Classification accuracy is given by average class accuracy and individual
        video accuracy.
        Face detection accuracy is given by precision and recall values.

        Args:
            cuda: A bool value that specifies if cuda shall be used
            detector: A string path to a .pth weights file for a face detection model
            classifier: A string path to a .pth weights file for a goggle classification model
            input_directory: Directory containing test videos to run Evaluator on
            annotation_path: Directory containing annotation files (output by annotator.py)
            rate: Run detection and classification on every 1/rate frames
        """

        if cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.device = torch.device('cuda:0')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            self.device = torch.device('cpu')

        if os.path.exists(DETECTIONS_FILE):
            os.remove(DETECTIONS_FILE)

        self.detector = FaceDetector(detector=detector, detector_type=detector_type, cuda=cuda and torch.cuda.is_available(),
                                     set_default_dev=True)
        # TODO check state_dict vs. not
        model = get_model()
        model.load_state_dict(torch.load(classifier, map_location=self.device))
        self.classifier = Classifier(model, self.device)
        #self.classifier = Classifier(torch.load(classifier, map_location=self.device), self.device)
        self.video_filenames = self.get_video_files(input_directory)
        self.results = {'Goggles':
                            {'average_class_accuracy': 0.0,
                             'number_of_videos': 0,
                             'individual_video_results': {}
                             },
                        'Glasses':
                            {'average_class_accuracy': 0.0,
                             'number_of_videos': 0,
                             'individual_video_results': {}
                             },
                        'Neither':
                            {'average_class_accuracy': 0.0,
                             'number_of_videos': 0,
                             'individual_video_results': {}
                             }
                        }
        self.class_label = ''
        self.condition = ''
        self.cap = ''
        self.video = ''
        self.video_len = 0
        self.rate = rate
        self.evaluate(annotation_path)

    def evaluate(self, annotation_path: str):
        """
        Evaluates (classification and detection) every video file in the input directory
        containing test videos and stores results in self.results.
        To understand the format of self.results dict, check the constructor

        Args:
            annotation_path - Directory containing all the annotations of face detections
        """
        total_videos_processed = 0
        for video_file in self.video_filenames:
            self.video = video_file
            print(f"Processing {self.video} ..., video {total_videos_processed}/{len(self.video_filenames)}")

            self.class_label = self.get_class_label()
            self.condition = self.get_condition()
            self.cap = cv2.VideoCapture(self.video)

            if self.cap.isOpened():
                classification_result = self.evaluate_classifications()  # Also contains boxes
                self.record_results(classification_result)
                total_videos_processed += 1
                print(f"{self.video} : Done")
            else:
                print(f"Unable to open video {self.video}")
                continue

        self.calculate_average_class_accuracy()

        # ------- classification ^^^ detection vvv

        # TODO why is this returning something
        # TODO make it an optional arg to evaluate face detection
        #detection_results = self.evaluate_detections(annotation_path, DETECTIONS_FILE)

        print(f"\n {total_videos_processed} videos processed!")

    def evaluate_classifications(self):
        """
        Returns the accuracy (percentage_of_correct_predictions) of the
        predictions for a video
        """
        inferences = self.infer()
        if sum(inferences.values()) == 0:
            percentage_of_correct_predictions = 0
        else:
            percentage_of_correct_predictions = inferences[self.class_label] / sum(inferences.values())

        return percentage_of_correct_predictions, inferences, sum(inferences.values())

    def evaluate_detections(self, ground_truth_detections_file, predicted_detections_file, overlap_threshold=0.5):
        """
        Calculates the recall and precision of face detection for a video.
        TODO explain what that means... seems like overlap of x and y coords? I.e. IoU?

        @param ground_truth_detections_file: file containing actual face detections (created by annotator.py)
        @param predicted_detections_file: file containing predicted face detections
        @param overlap_threshold: IoU greater than threshold counts as correct, less than is incorrect
        """

        with open(ground_truth_detections_file) as detect_file:
            ground_truth_detections = json.load(detect_file)

        with open(predicted_detections_file, 'r') as prediction_file:
            predicted_detections = json.load(prediction_file)

        # TODO fix below based on detections format
        total_ground_truths = 0
        for frame_id in ground_truth_detections:
            total_ground_truths += len(ground_truth_detections[frame_id])

        # TODO ugly parsing and such here. Need to debug it. ==1 means...?
        if any(predicted_detections) == 1:
            splitlines = [x.strip().split('|') for x in predicted_detections]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[5]) for x in splitlines])
            bboxes = np.array([[float(z) for z in x[1:5]] for x in splitlines])

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            bboxes = bboxes[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            nd = len(image_ids)
            true_pos = np.zeros(nd)
            false_pos = np.zeros(nd)

            # TODO for frame in frames?
            for d in range(nd):
                try:
                    bbox = bboxes[d, :].astype(float)
                    max_overlap = -np.inf
                    bbox_ground_truth_detections = np.asarray(ground_truth_detections[image_ids[d]], dtype=np.float32)
                    if bbox_ground_truth_detections.size > 0:
                        # TODO max and min variable names are backwards?
                        ixmin = np.maximum(bbox_ground_truth_detections[:, 0], bbox[0])
                        iymin = np.maximum(bbox_ground_truth_detections[:, 1], bbox[1])
                        ixmax = np.minimum(bbox_ground_truth_detections[:, 2], bbox[2])
                        iymax = np.minimum(bbox_ground_truth_detections[:, 3], bbox[3])
                        iw = np.maximum(ixmax - ixmin, 0.)
                        ih = np.maximum(iymax - iymin, 0.)
                        # TODO debug. inters = intersection? uni = union? Overlaps is actual value?
                        # TODO import IoU from box_utils should work
                        inters = iw * ih
                        uni = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) +
                               (bbox_ground_truth_detections[:, 2] - bbox_ground_truth_detections[:, 0]) *
                               (bbox_ground_truth_detections[:, 3] - bbox_ground_truth_detections[:, 1]) - inters)
                        overlaps = inters / uni
                        max_overlap = np.max(overlaps)
                        # jmax = np.argmax(overlaps)

                    if max_overlap > overlap_threshold:
                        true_pos[d] = 1.
                    else:
                        false_pos[d] = 1.

                except KeyError:
                    continue

            print("Total ground truths: ", total_ground_truths)
            false_pos = np.cumsum(false_pos)
            true_pos = np.cumsum(true_pos)
            recall = true_pos / float(total_ground_truths)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            precision = true_pos / np.maximum(true_pos + false_pos, np.finfo(np.float64).eps)
        else:
            recall = -1.
            precision = -1.
            ap = -1.

        print("Precision: ", precision)
        print("Recall: ", recall)

        return precision[len(precision)], recall[len(recall)]  # final precision, recall

    def calculate_average_class_accuracy(self):
        """
        Calculates the average class accuracy for each class and stores it in self.results
        """
        for class_label in self.results:
            if self.results[class_label]['number_of_videos'] > 0:
                self.results[class_label]['average_class_accuracy'] = self.results[class_label][
                                                                          'average_class_accuracy'] / \
                                                                      self.results[class_label]['number_of_videos']

    def record_results(self, result):
        """
        Records results of one video in the self.results dict

        @param result(List) - contains the classification accuracy,
        number of predictions for each label, number of detections
        """
        self.results[self.class_label]['number_of_videos'] += 1
        # below is just a running sum which gets divided by the number of videos at the end
        self.results[self.class_label]['average_class_accuracy'] += result[0]
        self.results[self.class_label]['individual_video_results'][self.video] = {}
        self.results[self.class_label]['individual_video_results'][self.video]["accuracy"] = result[0]
        self.results[self.class_label]['individual_video_results'][self.video]["glasses"] = result[1]['Glasses']
        self.results[self.class_label]['individual_video_results'][self.video]["goggles"] = result[1]['Goggles']
        self.results[self.class_label]['individual_video_results'][self.video]["neither"] = result[1]['Neither']
        self.results[self.class_label]['individual_video_results'][self.video]["num_detections"] = result[2]
        self.results[self.class_label]['individual_video_results'][self.video]["num_frames"] = self.video_len
        self.results[self.class_label]['individual_video_results'][self.video]["condition"] = self.condition

    def record_detections(self, file, detections):
        """
        Save face detections in a file for evaluation
        TODO improve how this is stored
        Args:
            file (str): Records detections here
            detections (List): contains all the bounding boxes and confidence values
        """
        f = open(file, "a+")
        for detection in detections:
            for element in detection:
                f.write(str(element))
                f.write("|")
            f.write("\n")
        f.close()

    def infer(self):
        """
        Performs inference on a video using the face detection
        and goggle classification models.
        @param rate: How often to run detection (every 1/rate frames).
        It returns:
        1) inference_dict: the number of inferences for each class.
        """
        bboxes = []
        preds = []
        inference_dict = {"Goggles": 0, "Glasses": 0, "Neither": 0}

        # check if the video needs to be rotated
        rotate_code = check_rotation(self.video)
        self.video_len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_num in tqdm(range(self.video_len)):
            ret, img = self.cap.read()
            if frame_num % self.rate == 0 and img is not None:
                if rotate_code is not None:
                    correct_rotation(img, rotate_code)
                frame_id = self.video.strip('.avi').strip('.mp4').strip('.MOV').strip('.mov').split('/')[-1] + "_" + str(
                    frame_num)
                boxes = self.detector.detect(img)  # Also contains confidence
                for box in boxes:
                    x1 = max(0, box[0])
                    y1 = max(0, box[1])
                    x2 = min(img.shape[1], box[2])
                    y2 = min(img.shape[0], box[3])
                    conf = box[4]
                    face = img[int(y1):int(y2), int(x1):int(x2), :]
                    label = self.classifier.classifyFace(face)
                    preds.append(label.item())
                    bboxes.append([frame_id, x1, y1, x2, y2, conf])

        inference_dict["Glasses"] += preds.count(0)
        inference_dict["Goggles"] += preds.count(1)
        inference_dict["Neither"] += preds.count(2)

        self.record_detections(DETECTIONS_FILE, bboxes)
        return inference_dict

    def get_class_label(self):
        """
        Get class label [Goggles / Glasses / Neither] that the image belongs to
        """
        if '/Goggles/' in self.video or '/goggles/' in self.video:
            class_label = 'Goggles'
        elif '/Glasses/' in self.video or '/glasses/' in self.video:
            class_label = 'Glasses'
        else:
            class_label = 'Neither'

        return class_label

    def get_condition(self):
        """
        Get condition [Ideal, low_lighting etc. ] that the image belongs to
        """
        return self.video.split('/')[-2]

    def get_video_files(self, input_directory: str):
        """
        Gets all the video files in the input directory
        """
        filenames = []
        for dirName, subdirList, fileList in os.walk(input_directory):
            for filename in fileList:
                ext = '.' + filename.split('.')[-1]
                if ext in VIDEO_EXT:
                    filenames.append(dirName + '/' + filename)

        return filenames

    def get_evaluator_results(self):
        """
        Returns the dict containing all the test results (self.results)
        """
        return self.results


if __name__ == "__main__":
    warnings.filterwarnings("once")
    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument('--detector', '-d', type=str, default='model_weights/blazeface.pth',
                        help="Path to a trained face detector .pth file")
    parser.add_argument('--detector_type', '-t', type=str, help="One of blazeface, retinaface, ssd")
    parser.add_argument('--classifier', default='model_weights/ensemble_100epochs.pth', type=str,
                        help="Path to a trained classifier .pth file")
    parser.add_argument('--cuda', '-c', default=False, action='store_true', help="Enable CUDA")
    parser.add_argument('--input_directory', type=str, required=True, help="Path to a directory containing video files")
    parser.add_argument('--annotation_path', type=str, required=True, help="Path to a directory containing annotation "
                                                                           "files")
    parser.add_argument('--rate', '-r', type=int, default=1, help='Run detection on every 1/rate frames.')
    # TODO add store_true args for detection, evaluation (to do separately if desired)

    args = parser.parse_args()

    if not args.input_directory:
        raise Exception("Invalid input directory")
    evaluator = Evaluator(args.cuda, args.detector, args.detector_type, args.classifier, args.input_directory,
                          args.annotation_path)
    individual_video_results = evaluator.get_evaluator_results()

    with open(CLASSIFICATION_RESULTS_FILE, 'w+') as json_file:
        json.dump(individual_video_results, json_file, indent=4)

    print(f"\n Output saved at {CLASSIFICATION_RESULTS_FILE}")

    exit()
