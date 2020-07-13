from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
import json

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.5, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=1000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.05, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=250, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--output_directory', default='ground_truth_detections_lowlight/', type=str, help='directory to store detected labels')
parser.add_argument('--input_directory', default='test_videos/', type=str, help='directory where test videos are located')


args = parser.parse_args()
CLASSES = ['Glasses/', 'Goggles/', 'Neither/']
CONDITIONS = ['Ideal/', 'Low_lighting/', 'Occlusion_bottom/', 'Occlusion_left_right/', 'Pose_45_degrees_down/', 'Pose_45_degrees_up/',
            'Pose_looking_left/', 'Pose_looking_right/', 'Scale_3-5m/', 'Scale_<3m/', 'Scale_>5m/']

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def create_directories(root_directory, create_class_condition_directories=False):
    if not os.path.isdir(root_directory):
        os.mkdir(root_directory)



def get_storage_location(output_directory, video_filename, input_directory):
    save_dir = output_directory + video_filename.strip(input_directory).strip('.mp4').strip('.mov').strip('.MOV').strip('.avi').split('/')[-1] + '_'
    #create_directories(save_dir)

    return save_dir



def get_videos(input_directory):
    filenames = []
    for dirName, subdirList, fileList in os.walk(input_directory):
        for filename in fileList:
            ext = '.' + filename.split('.')[-1]
            if ext in ['.mov','.mp4','.avi', '.MOV']:
                filenames.append(dirName + '/' + filename)

    return filenames


if __name__ == '__main__':
    create_directories(args.output_directory, create_class_condition_directories=True)

    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 0.4

    video_files = get_videos(args.input_directory)

    for video in video_files:
        cap = cv2.VideoCapture(video)
        storage_location = get_storage_location(args.output_directory, video, args.input_directory)
        create_directories(storage_location)
        print ("Video: ", video)


        # testing begin
        if cap.isOpened():
            frame_number = 0
            while True:
                ret, img_raw = cap.read()
                if not ret:
                    break
                img = np.float32(img_raw)
                img = cv2.resize(img, (int(img.shape[1]*resize), int(img.shape[0]*resize)))

                im_height, im_width, _ = img.shape
                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                img -= (104, 117, 123)
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).unsqueeze(0)
                img = img.to(device)
                scale = scale.to(device)

                tic = time.time()
                loc, conf, landms = net(img)  # forward pass
                #print('net forward time: {:.4f}'.format(time.time() - tic))

                priorbox = PriorBox(cfg, image_size=(im_height, im_width))
                priors = priorbox.forward()
                priors = priors.to(device)
                prior_data = priors.data
                boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
                boxes = boxes * scale / resize
                boxes = boxes.cpu().numpy()
                scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
                landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
                scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                       img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                       img.shape[3], img.shape[2]])
                scale1 = scale1.to(device)
                landms = landms * scale1 / resize
                landms = landms.cpu().numpy()

                # ignore low scores
                inds = np.where(scores > args.confidence_threshold)[0]
                boxes = boxes[inds]
                landms = landms[inds]
                scores = scores[inds]

                # keep top-K before NMS
                order = scores.argsort()[::-1][:args.top_k]
                boxes = boxes[order]
                landms = landms[order]
                scores = scores[order]

                # do NMS
                dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = py_cpu_nms(dets, args.nms_threshold)
                # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
                dets = dets[keep, :]
                landms = landms[keep]

                # keep top-K faster NMS
                dets = dets[:args.keep_top_k, :]
                landms = landms[:args.keep_top_k, :]

                #dets = np.concatenate((dets, landms), axis=1)
                output_file = storage_location + str(frame_number) + '.txt'
                f = open(output_file, "w")
                for detection in dets:
                    for coord in detection:
                        f.write(str(coord) + " ")
                    f.write("\n")
                f.close()

                frame_number += 1

    exit(0)
