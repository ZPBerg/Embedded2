import cv2
import ffmpeg

"""
Miscellaneous utility functions that apply to multiple scripts.
"""


"""
check_rotation and correct_rotation adapted from 
https://stackoverflow.com/questions/53097092/frame-from-video-is-upside-down-after-extracting
to handle the fact that some videos store rotation metadata while others do not,
and OpenCV can't tell the difference.
"""


def check_rotation(path_video_file: str):
    # only .mov files need to be rotated
    if path_video_file.split('.')[-1].upper() != '.MOV':
        return None

    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotate_code = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotate_code = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotate_code = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotate_code


def correct_rotation(frame, rotate_code):
    return cv2.rotate(frame, rotate_code)


def bbox_iou(boxA, boxB):
    """
    Calculate IoU (Intersection over Union) of two bounding boxes.
    @param boxA: the top left and bottom right coords of the box
    as a list [xmin, ymin, xmax, ymax]
    @param boxB: the other box, same format as boxA.
    It doesn't matter which one is the ground truth bounding box.
    """

    for i in range(len(boxA)):
        boxA[i] = float(boxA[i])
        boxB[i] = float(boxB[i])

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    xA += 5

    # compute the area of intersection rectangle
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou