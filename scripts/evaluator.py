import argparse
import json
import os
import time
import warnings

import cv2
import numpy as np
import torch

from scripts.utils import check_rotation, correct_rotation
from src.jetson.main import FaceDetector, Classifier

VIDEO_EXT = ['.mov', '.mp4', '.avi', '.MOV', '.MP4', '.AVI']

"""
Use this script with annotator.py. 
Videos to be evaluated should be from the TestVideos folder on the Drive.
"""

# TODO - TODO TODO don't do face detection? Would have to manually label faces but we're using a
# TODO - SOTA face detection model that could just empirically be observed to work


class Evaluator():
    def __init__(self, cuda, detector, detector_type, classifier, input_directory, annotation_path):
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
        """

        if cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.device = torch.device('cuda:0')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            self.device = torch.device('cpu')

        if os.path.exists("det_results.txt"):
            os.remove("det_results.txt")

        self.detector = FaceDetector(detector=detector, detector_type=detector_type, cuda=cuda and torch.cuda.is_available(),
                                     set_default_dev=True)
        self.classifier = Classifier(torch.load(classifier, map_location=self.device), self.device)
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
        self.evaluate(annotation_path)

    def evaluate(self, annotation_path: str):
        """
        Evaluates every video file in the input directory containing test videos and
        stores results in self.results.
        To understand the format of self.results dict, check the constructor

        Args:
            annotation_path - Directory containing all the annotations of face detections
        """
        total_videos_processed = 0
        for video_file in self.video_filenames:
            self.video = video_file
            print(f"Processing {self.video} ...")

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
        #detection_results = self.evaluate_detections(annotation_path, "det_results.txt")

        print(f"\n {total_videos_processed} videos processed!")

    def calculate_average_class_accuracy(self):
        """
        Calculates the average class accuracy for each class and stores it in the
        self.results dict.
        """
        for class_label in self.results:
            if self.results[class_label]['number_of_videos'] > 0:
                self.results[class_label]['average_class_accuracy'] = self.results[class_label][
                                                                          'average_class_accuracy'] / \
                                                                      self.results[class_label]['number_of_videos']

    def record_results(self, result):
        """
        Records all the results in the self.results dict

        Args:
            result(List) - contains the classification accuracy and inference time
        """
        self.results[self.class_label]['number_of_videos'] += 1
        self.results[self.class_label]['average_class_accuracy'] += result[0]
        self.results[self.class_label]['individual_video_results'][self.video] = {}
        self.results[self.class_label]['individual_video_results'][self.video]["accuracy"] = result[0]
        self.results[self.class_label]['individual_video_results'][self.video]["inference_time"] = result[1]
        self.results[self.class_label]['individual_video_results'][self.video]["condition"] = self.condition

    def record_detections(self, file, detections):
        """
        Save detections in a file for evaluation
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
        Performs inference on a video by using the face detection
        and goggle classification models
        It returns:
        1) inference_dict: the number of inferences for each class.
        2) average_inference_time: a float containing the average inference time for the whole video
        """
        bboxes = []
        preds = []
        inference_dict = {"Goggles": 0, "Glasses": 0, "Neither": 0}
        frame_counter = 0
        start_time = time.time()

        # check if the video needs to be rotated
        rotate_code = check_rotation(self.video)

        while True:
            ret, img = self.cap.read()
            if not ret:
                break
            if rotate_code is not None:
                correct_rotation(img, rotate_code)
            # img = cv2.resize(img, (640, 480))  #Set this to the input shape of image for faster processing. (Remember to do the same in annotator)
            frame_id = self.video.strip('.avi').strip('.mp4').strip('.MOV').strip('.mov').split('/')[-1] + "_" + str(
                frame_counter)
            boxes = self.detector.detect(img)  # Also contains confidence
            box_no_conf = []
            if len(boxes) != 0:
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

                    inference_dict["Goggles"] += preds.count(1)
                    inference_dict["Glasses"] += preds.count(0)
                    inference_dict["Neither"] += preds.count(2)

        total_time = time.time() - start_time
        if frame_counter > 0:
            average_inference_time = total_time / frame_counter
        else:
            average_inference_time = -1  # Empty video file

        # TODO make det_results.txt a global variable DETECTION_FILE
        self.record_detections("det_results.txt", bboxes)
        return inference_dict, average_inference_time

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

    def get_ground_truth_detections(self, directory):
        """
        Get ground truth detection labels (from annotation file)
        """
        ground_truths = {}

        for file in os.listdir(directory):
            f = open(directory + file, "r")
            key = file.strip('.txt')
            content = f.readlines()
            f.close()

            content = [list(map(float, x.strip(' \n').split(' '))) for x in content]
            ground_truths[key] = content

        return ground_truths

    def evaluate_classifications(self):
        """
        Returns the accuracy (percentage_of_correct_predictions) of the
        predictions for a video
        """
        inferences, inference_time = self.infer()
        if sum(inferences.values()) == 0:
            percentage_of_correct_predictions = 0
        else:
            percentage_of_correct_predictions = inferences[self.class_label] / sum(inferences.values())

        return percentage_of_correct_predictions, inference_time

    def evaluate_detections(self, annotations_dir, detection_dir, overlap_threshold=0.5):
        """
        Calculates the recall and precision of face detection for a video.
        TODO explain what that means... seems like overlap of x and y coords? I.e. IoU?
        
        @param annotations_dir: directory containing annotation files (created by annotator.py)
        @param detection_dir: directory of predicted detections TODO ???
        @param overlap_threshold: greater than threshold counts as correct, less than is incorrect
        """

        ground_truth_detections = self.get_ground_truth_detections(annotations_dir)
        with open(detection_dir, 'r') as f:
            # TODO verify variable name accurate
            predicted_detections = f.readlines()

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


def main():
    if not args.input_directory:
        raise Exception("Invalid input directory")
    evaluator = Evaluator(args.cuda, args.detector, args.detector_type, args.classifier, args.input_directory, args.annotation_path)
    individual_video_results = evaluator.get_evaluator_results()

    with open(args.output_file, 'w+') as json_file:
        json.dump(individual_video_results, json_file, indent=4)

    print(f"\n Output saved at {args.output_file}")


if __name__ == "__main__":
    warnings.filterwarnings("once")
    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument('--detector', '-d', type=str, default='model_weights/blazeface.pth',
                        help="Path to a trained face detector .pth file")
    parser.add_argument('--detector_type', '-t', type=str, help="One of blazeface, retinaface, ssd")
    parser.add_argument('--classifier', default='model_weights/ensemble_100epochs.pth', type=str,
                        help="Path to a trained classifier .pth file")
    parser.add_argument('--cuda', '-c', default=False, action='store_true', help="Enable CUDA")
    parser.add_argument('--output_file', type=str, default='results.json',
                        help="Name of evaluation log")
    parser.add_argument('--input_directory', type=str, required=True, help="Path to a directory containing video files")
    parser.add_argument('--annotation_path', type=str, required=True, help="Path to a directory containing annotation "
                                                                           "files")
    # TODO add store_true args for detection, evaluation (to do separately if desired)

    args = parser.parse_args()

    main()

    exit()
