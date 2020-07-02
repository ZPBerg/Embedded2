import argparse
import json

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

from goggle_classifier import train_model

# 80/20 training/validation split
VAL_SPLIT = 0.2

"""Train a custom CNN classifier trained on activation maps output by extract_features.py"""


def tensor_loader(path):
    """Required for the custom dataset created in load_data."""
    return torch.load(path)


def load_data(data_location):
    """
    Create a Pytorch DataLoader for activation maps output by the face detector.
    This is quite similar to goggle_classifier's load_data method, but no transforms are required here.
    @param data_location: Directory in DatasetFolder structure containing .pt files to train on.
    @return: The Pytorch Dataloader, size of training and validation datasets, and dataset class names.
    """
    dataset = datasets.DatasetFolder(data_location, tensor_loader, ('.pt',))

    val_size = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    splits = torch.utils.data.random_split(dataset, [train_size, val_size])
    tensor_datasets = {'train': splits[0],
                       'val': splits[1]}

    # TODO batch_size could be 4... need to understand how transforms are handled
    data_loaders = {'train': DataLoader(tensor_datasets['train'], batch_size=1,
                                        shuffle=True, num_workers=4),
                    'val': DataLoader(tensor_datasets['val'], batch_size=1,
                                      shuffle=True, num_workers=4)}
    dataset_sizes = {x: len(tensor_datasets[x]) for x in ['train', 'val']}

    return data_loaders, dataset_sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run classification on a dataset')
    parser.add_argument('--directory', type=str, help='Relative directory location of dataset', default='dataset')
    args = parser.parse_args()

    # TensorBoard writer
    writer = SummaryWriter()

    # the CNN architecture to train on. The exact layers involved is an open question.
    model = nn.Sequential(
        # One or more convolutional layers here?
        nn.Flatten(),
        nn.Dropout(0.2),
        # 6144 is 96 x 8 x 8 (size of the activation map)
        nn.Linear(6144, 1000),
        # 3 is the number of classes
        nn.Linear(1000, 3)
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with open("act_map_params.json", "r") as params_file:
        params = json.load(params_file)

    data_loaders, dataset_sizes = load_data(args.directory)
    trained_model = train_model(model, data_loaders, dataset_sizes, params)
    torch.save(trained_model, 'act_map_model.pth')

    exit(0)
