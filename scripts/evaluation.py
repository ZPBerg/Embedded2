import argparse
import json
import warnings

import torch

from scripts.evaluator import Evaluator

PRED_DETECTIONS_FILE = 'detection_predictions.csv'
CLASSIFICATION_RESULTS_FILE = 'results.json'

"""
Evaluate classification and (optionally) compare face detection models on a set of videos.
Videos to be evaluated should be from the TestVideos folder on the Drive 
to get correct labels and conditions.
To compare face detection models, run annotator.py first.
"""

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
    parser.add_argument('--detection_file', type=str, help="Path to the detections csv output by annotator.py."
                                                           "If given, the detections will be compared.")
    parser.add_argument('--rate', '-r', type=int, default=1, help='Run detection on every 1/rate frames.')

    args = parser.parse_args()

    if not args.input_directory:
        raise Exception("Invalid input directory")

    evaluator = Evaluator(args.cuda and torch.cuda.is_available(), args.detector, args.detector_type, args.classifier,
                          args.input_directory, args.rate, args.detection_file, PRED_DETECTIONS_FILE)
    individual_video_results = evaluator.get_evaluator_results()

    with open(CLASSIFICATION_RESULTS_FILE, 'w+') as json_file:
        json.dump(individual_video_results, json_file, indent=4)

    print(f"\n Output saved at {CLASSIFICATION_RESULTS_FILE}")

    exit()
