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
    if path_video_file.split('.')[-1] != '.MOV' or '.mov':
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
