import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from .net import MobileNetV1 as MobileNetV1
from .net import FPN as FPN
from .net import SSH as SSH


class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        '''
        Adds layers on top of feature extractor for classification

        Args:
            inchannels(int) - Number of input channels
            num_anchors(int) - Number of anchor boxes
        '''
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x:torch.Tensor):
        """Applies network layers and ops on input tensor x

        Args:
            x: tensor outputed by the feature extractor

        Returns reshaped output tensor after passing through a 1x1 conv layer
        """
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        '''
        Adds layers on top of feature extractor for finding face coordinates

        Args:
            inchannels(int) - Number of input channels
            num_anchors(int) - Number of anchor boxes
        '''
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x:torch.Tensor):
        """Applies network layers and ops on input tensor x.

        Args:
            x: tensor outputed by the feature extractor

        Returns reshaped output tensor after passing through a 1x1 conv layer
        """
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        '''
        Adds layers on top of feature extractor for finding face landmark coordinates

        Args:
            inchannels(int) - Number of input channels
            num_anchors(int) - Number of anchor boxes
        '''
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x:torch.Tensor):
        """Applies network layers and ops on input tensor x.

        Args:
            x: tensor outputed by the feature extractor

        Returns reshaped output tensor after passing through a 1x1 conv layer
        """
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        Retinaface model used for face face detection

        Args:
            cfg (dict):  Network related settings.
            phase (string): train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None

        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()

            if cfg['pretrain']:
                #Load mobilenet model with pretrained weights at model_weights/mobilenetV1X0.25_pretrain.tar
                checkpoint = torch.load("model_weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])


        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        '''
        Add layer on top of retinaface for classification (face or not)

        Args:
            fpn_num(int) - Number of feature pyramid network layers
            inchannels(int) - Number of input channels
            anchor_num(int) - Number of anchors

        '''
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead

    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        '''
        Add layer on top of retinaface for outputing bounding box coordinates

        Args:
            fpn_num(int) - Number of feature pyramid network layers
            inchannels(int) - Number of input channels
            anchor_num(int) - Number of anchors

        '''
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        '''
        Add layer on top of retinaface for outputing facial landmark coordinates

        Args:
            fpn_num(int) - Number of feature pyramid network layers
            inchannels(int) - Number of input channels
            anchor_num(int) - Number of anchors

        '''
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs:torch.Tensor):
        """Applies network layers and ops on input image(s)

        Args:
            inputs: input image or batch of images.

        Return:
            output (tuple):
                bbox_regressions - bounding box coordinates
                classifications - class confidences (F.softmax done during test phase to output probability of each class)
                ldm_regressions - face landmark coordinates

        """
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)

        return output


def load_model(model:'RetinaFace Object', pretrained_path:str, load_to_cpu:bool):
    '''
    Load retinaface model
    Args:
        model: Model to load
        pretrained_path: Contains location of pretrained model weights
        load_to_cpu: Use CPU for inference

    Returns trained model loaded to desired device
    '''
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    model.load_state_dict(pretrained_dict, strict=False)
    return model
