import time
import numpy as np
import mock
import random
from mock import patch

from src.jetson.video_capturer import VideoCapturer
def mock_read():
    img = np.zeros((300, 300, 3), np.uint8)
    img[:] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    return None, img

@mock.patch('cv2.VideoCapture')
class TestCapturer():
    '''
    Tests in this class are for the VideoCapturer class found in src/video_capturer.py
    '''

    def test_init(self, mock_capture):
        mock_capture.return_value.read.return_value = mock_read() 
        self.capturer = VideoCapturer(False)
        '''
        Tests constructor
        Checks:
            - Update thread is created
            - Capturer is not null (interface with camera established)
            - First captured frame is not null
        '''
        assert self.capturer.t1.isAlive()
        assert self.capturer.capture is not None
        assert self.capturer.frame is not None

    def test_update(self, mock_capture):
        mock_capture.return_value.read.return_value = mock_read()
        self.capturer = VideoCapturer(False)
        '''
        Tests 'update' function
        Checks:
            - Frames captured are not null
            - Every frame is different from the previous frame
        '''
        assert self.capturer.frame is not None
        oldFrame = self.capturer.frame.copy()
        time.sleep(.02) # sleep long enough to ensure thread t1 updates
        newFrame = self.capturer.frame.any()
        compare = oldFrame == newFrame
        assert not compare.all()

    def test_get_frame(self, mock_capture):
        mock_capture.return_value.read.return_value = mock_read()
        self.capturer = VideoCapturer(False)
        '''
        Tests 'get_frame' function
        Checks:
            - Frame returned is not null
            - Frame returned is equal to frame instance variable
        '''
        assert self.capturer.frame is not None
        assert self.capturer.frame is self.capturer.get_frame()

    def test_close(self, mock_capture):
        mock_capture.return_value.read.return_value = mock_read()
        self.capturer = VideoCapturer(False)
        '''
        Tests 'close' function
        Checks:
            - Update thread is no longer alive
        '''
        self.capturer.close()
        assert self.capturer.t1.isAlive() == False
        
        
