import numpy as np
import torch

from src.jetson.models.Retinaface.data.config import cfg_mnet as cfg
from src.jetson.models.Retinaface.data.config import cfg_inference as infer_params
from src.jetson.models.utils.transform import BaseTransform
from src.jetson.models.Retinaface.layers.functions.prior_box import PriorBox
from src.jetson.models.utils.box_utils import decode, do_nms, postprocess

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
            self.net.load_state_dict(torch.load(detector, map_location=self.device))
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
            self.net = load_model(self.net, detector, load_to_cpu=self.device == torch.device("cpu"))
            self.model_name = 'retinaface'
            self.image_shape = infer_params["image_shape"]  # (H, W)
            self.resize = infer_params["resize"]
            self.transformer = BaseTransform((self.image_shape[1], self.image_shape[0]), (104, 117, 123))
            priorbox = PriorBox(cfg, image_size=self.image_shape)
            priors = priorbox.forward()
            self.prior_data = priors.data.to(self.device)

        self.detection_threshold = detection_threshold
        self.net.to(self.device)
        self.net.eval()

    def detect(self,
               frame: np.ndarray):
        """
        Performs face detection on the frame passed
        Args:
            frame: A 3D numpy array representing an image

        Return:
            The bounding boxes of the face(s) that were detected formatted (upper left corner(x, y) , lower right corner(x,y))
        """

        if self.model_name == 'ssd':
            x = torch.from_numpy(self.transformer(frame)[0]).permute(2, 0, 1)
            x = Variable(x.unsqueeze(0)).to(self.device)
            y = self.net(x)
            detections = y.data
            scale = torch.Tensor([frame.shape[1], image.shape[0], image.shape[1], image.shape[0]])
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
            transformed_frame = self.transformer(frame)[0].astype(np.float32)

            detections = self.net.predict_on_frame(transformed_frame)
            if isinstance(detections, torch.Tensor):
                detections = detections.cpu().numpy()

            if detections.ndim == 1:
                detections = np.expand_dims(detections, axis=0)

            bboxes = []
            for i in range(detections.shape[0]):
                ymin = detections[i, 0] * frame.shape[0]
                xmin = detections[i, 1] * frame.shape[1]
                ymax = detections[i, 2] * frame.shape[0]
                xmax = detections[i, 3] * frame.shape[1]
                conf = detections[i, 16]

                transformed_frame = img / 127.5 - 1.0

                for k in range(6):
                    kp_x = detections[i, 4 + k * 2] * transformed_frame.shape[1]
                    kp_y = detections[i, 4 + k * 2 + 1] * transformed_frame.shape[0]

                bboxes.append((xmin, ymin, xmax, ymax, conf))

            return bboxes

        elif self.model_name == 'retinaface':
            transformed_frame = (self.transformer(frame)[0]).transpose(2, 0, 1)
            transformed_frame = torch.from_numpy(transformed_frame).unsqueeze(0)
            transformed_frame = transformed_frame.to(self.device)
            loc, conf, _ = self.net(
                transformed_frame)  # forward pass: Returns bounding box location, confidence and facial landmark locations

            boxes = decode(loc.data.squeeze(0), self.prior_data, cfg['variance'])
            boxes, scores = postprocess(boxes, conf, self.image_shape, self.detection_threshold, self.resize)
            dets = do_nms(boxes, scores, infer_params["nms_thresh"])

            #scale box coordinates back to original frame dimensions
            for det in dets:
                det[0] *= frame.shape[1] / transformed_frame.shape[3]
                det[1] *= frame.shape[0] / transformed_frame.shape[2]
                det[2] *= frame.shape[1] / transformed_frame.shape[3]
                det[3] *= frame.shape[0] / transformed_frame.shape[2]

            bboxes = [tuple(det[0:5]) for det in dets]

            return bboxes
