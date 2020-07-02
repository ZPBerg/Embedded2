from __future__ import print_function, division

import argparse

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch import multiprocessing

from goggle_classifier import load_data, train_model
from Embedded2.src.jetson.main import FaceDetector

VAL_SPLIT = .2
activation = {}

"""Save activation maps of images run partway through a CNN. Currently specific to Blazeface. 
Use this script before running learn_features.py"""

"""Transformations to perform on images before running inference. Blazeface requires 128x128 images."""
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ]),
}


# custom dataset for applying different transforms to train and val data
class MapDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, item):
        return self.map(self.dataset[item][0]), self.dataset[item][1]

    def __len__(self):
        return len(self.dataset)


def run_inference(detector, data_location, data_loaders,  class_names):
    """
    Run all images in the dataset through the face detector, saving activation maps as .pt files.
    @param detector: The FaceDetector. Must use a Blazeface model for now.
    @param data_location: Directory in Imagefolder structure containing the images to train on.
    @param data_loaders: A dictionary containing a 'train' DataLoader
    and a 'val' DataLoader (returned by load_data).
    @param class_names: List of class names. Used for saving .pt files in the same folder as their respective image.
    """

    i = 0
    for phase in ['train', 'val']:
        # iterate over all data
        for inputs, labels in data_loaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # TODO blazeface never appears to detect any faces...
            # the activation map is saved when .detect is called, so we need to call this even though it's not used
            output = detector.detect(inputs)

            # TODO I think there's a nicer way to write this file name, something like os.path.join?
            # preferably should/could have the same name as the original image, but with a .pt extension.
            # does the name get ignored when loading though?
            torch.save(activation['backbone2'],
                       data_location + '/' + class_names[labels.item()] + '/act_map' + str(i) + '.pt')
            i += 1


def get_activation(name):
    # TODO find reference for why this is the correct way
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run classification on a dataset')
    parser.add_argument('--directory', '-d', type=str, required=True, help='(Relative) Directory location of dataset')
    parser.add_argument('--cuda', '-c', default=False, action='store_true', help="Enable cuda")

    args = parser.parse_args()

    # TODO I think this is necessary when using CUDA b/c we have two processes running?
    multiprocessing.set_start_method('spawn')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # this is hard-coded for Blazeface because it is the only supported model
    # there are no current plans to integrate other models
    detector = FaceDetector(trained_model='blazeface.pth', cuda=args.cuda and torch.cuda.is_available(),
                            set_default_dev=True)

    # saves activation map after backbone2 into activation['backbone2']
    detector.net.backbone2.register_forward_hook(get_activation('backbone2'))

    data_loaders, dataset_sizes, class_names = load_data(args.directory, data_transforms)

    run_inference(detector, args.directory, data_loaders, class_names)

    exit(0)
