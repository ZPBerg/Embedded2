import cv2
from multiprocessing import Process, Queue, Value
from threading import Thread
import time

def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=820,
    display_height=616,
    framerate=21,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


class VideoCapturer(object):
    def __init__(self, gstreamer, dev=1):
        """
        This class captures videos using open-cv's VideoCapture object
        Args:
            dev: ID of mounted video device to be used for video capture (default is 0)
            gstreamer: Bool that states whether or not gstreamer pipeline should be crated (for pi camera)
        """
        if gstreamer:
            self.capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        else:
            self.capture = cv2.VideoCapture(dev)

        _, self.frame = self.capture.read()
        self.running = True
        self.t1 = Thread(target=self.update, args=())
        self.t1.daemon = True
        self.t1.start()

    def update(self):
        """Get next frame in video stream"""
        while self.running:
            if self.capture.isOpened():
                _, self.frame = self.capture.read()
            time.sleep(.01)

    def get_frame(self):
        """ Return current frame in video stream"""
        return self.frame

    def close(self):
        self.running = False
        self.t1.join()
