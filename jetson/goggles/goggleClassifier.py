from __future__ import print_function, division

import argparse
import json
import os
import time
import warnings

import matplotlib.pyplot as plt
import prettytable as pt
import sklearn.metrics as skm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets, models


class MapDataset(torch.utils.data.Dataset):
    """Custom dataset for applying different transforms to training and validation data."""
    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, item):
        return self.map(self.dataset[item][0]), self.dataset[item][1]

    def __len__(self):
        return len(self.dataset)


"""Dictionary of data augmentations during training and validation."""
data_transforms = {
    'trainaug1': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        # can't do rotations + grayscale in torchvision 0.5.0: https://github.com/pytorch/vision/issues/1759
        # if we decide rotation invariance + grayscale is necessary we'll need to upgrade to 0.6.0
        transforms.RandomCrop(224, pad_if_needed=True),
        transforms.Grayscale,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'trainaug2': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(224, pad_if_needed=True),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'trainaug3': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.5, 0.5),
        transforms.RandomCrop(224, pad_if_needed=True),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valaug1': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valaug2': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

"""
Converts the layer number to the number of parameters until that layer.
There are in total 19 (0 - 18) layers in mobilenet (a combination of convolutional layers and inverted
residual layers.)
The key for not freezing any layers is '-1'.
"""
mobilenet_v2_layer_parameter = {'-1': 0, '0': 3, '1': 9, '2': 18, '3': 27, '4': 36, '5': 45, '6': 54,
                                '7': 63, '8': 72, '9': 81, '10': 90, '11': 99, '12': 108, '13': 117,
                                '14': 126, '15': 135, '16': 144, '17': 153, '18': 162}


def get_model(last_layer_to_freeze='-1'):
    """Initialize Mobilenet, freezing relevant layers."""
    model = models.mobilenet_v2(pretrained=True)

    # Get and freeze parameters of the model
    parameters_to_freeze = mobilenet_v2_layer_parameter[last_layer_to_freeze]

    ctr = 0
    for name, param in model.features.named_parameters():
        if ctr < parameters_to_freeze:
            param.requires_grad = False
        ctr += 1

    param_ctr = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_ctr += 1

    print("Total parameters to be trained:", param_ctr)

    # Make a new classifier layer to match the number of classes
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, NUM_CLASSES)
    )

    return model


def load_data(data_location):
    """Create a Pytorch Dataloader for the images specified by args.directory."""
    dataset = datasets.ImageFolder(os.path.abspath(data_location))
    class_names = dataset.classes

    val_size = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    face_datasets = {}
    face_datasets['train'], face_datasets['val'] = torch.utils.data.random_split(dataset, [train_size, val_size])

    face_datasets['train'] = MapDataset(face_datasets['train'], data_transforms[train_aug])
    face_datasets['val'] = MapDataset(face_datasets['val'], data_transforms[val_aug])

    data_loaders = {'train': DataLoader(face_datasets['train'], batch_size=4,
                                        shuffle=True, num_workers=4),
                    'val': DataLoader(face_datasets['val'], batch_size=4,
                                      shuffle=True, num_workers=4)}
    dataset_sizes = {x: len(face_datasets[x]) for x in ['train', 'val']}

    print('class_names are {}'.format(class_names))
    return data_loaders, dataset_sizes, dataset.classes


def train_model(model):
    """Train model on dataset using hyperparameters from params.json"""

    # hyperparameters for training the model
    lr = params['lr']
    momentum = params['momentum']
    step_size = params['step_size']
    gamma = params['gamma']
    num_epochs = params['num_epochs']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    print('Running model with lr={}, momentum={}, step_size={}, gamma={}, num_epochs={}\n'.format(lr, momentum,
                                                                                                  step_size,
                                                                                                  gamma,
                                                                                                  num_epochs))

    since = time.time()
    epoch_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # train and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward propagation
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # print('Outputs are {}'.format(outputs))
                    # print('Labels are {}'.format(labels))

                    loss = criterion(outputs, labels)

                    # backprop
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                scheduler.step()
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
            else:
                writer.add_scalar('Loss/val', epoch_loss, epoch)
                writer.add_scalar('Accuracy/val', epoch_acc, epoch)

            # Save checkpoints every 10 epochs
            if epoch != 0 and epoch % 10 == 0:
                print('Saving state, epoch:', epoch)
                torch.save(model, repr(epoch) + '.pth')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s \n'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Final val Acc: {:4f}'.format(epoch_acc))

    return model


def get_metrics():
    """Output statistics from final epoch of training,
    including precision, recall, and the confusion matrix."""
    model.eval()
    full_correct = []
    full_pred = []

    # collect true and predicted labels for sklearn
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                full_correct.append(labels[j].item())
                full_pred.append(preds[j].item())

    acc = skm.accuracy_score(full_correct, full_pred)
    fone_score = skm.f1_score(full_correct, full_pred, average="weighted")
    precision = skm.precision_score(full_correct, full_pred, average="weighted")
    recall = skm.recall_score(full_correct, full_pred, average="weighted")
    cm = skm.confusion_matrix(full_correct, full_pred)

    print('F1-score: {}'.format(fone_score))
    print('Precision: {}'.format(precision))
    print('Recall: {}\n'.format(recall))

    writer.add_hparams(params, {'hparam/accuracy': acc, 'hparam/f1_score': fone_score,
                                'hparam/precision': precision, 'hparam/recall': recall})
    writer.flush()

    # this may be nice to put into Tensorboard at some point
    print("------------------Confusion matrix------------------")
    x = pt.PrettyTable()
    x.add_column('', class_names)
    for i in range(NUM_CLASSES):
        col = []
        for j in range(NUM_CLASSES):
            col.append(cm[j][i])
        x.add_column(class_names[i], col)

    print(x)
    print('Columns are actual labels, rows are predicted labels\n\n')


if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser(description='Train a Mobilenet classifier on a set of images.')
    parser.add_argument('--directory', type=str, help='Relative directory location of dataset in Imagefolder '
                                                      'structure.')
    parser.add_argument('--frozen', type=str, help='Last layer to freeze in mobilenet-v2 model. Total 19 layers ('
                                                   '0-18), -1 signifies no frozen layers', default='-1')
    parser.add_argument('--aug', type=int, help='Augmentations to train on. 1: Grayscale, 2: Random cropping and '
                                                'rotations, 3: Cropping, rotations, brightness, and contrast.', default=3)
    parser.add_argument('--model', type=str, help='Relative location of a model to load. If given, training will '
                                                  'start from this point.', default=None)
    args = parser.parse_args()

    # 3 classes, 80/20 training/validation split
    NUM_CLASSES = 3
    VAL_SPLIT = .2

    # TensorBoard writer
    writer = SummaryWriter()

    print("Time: {}".format(time.asctime(time.localtime())))

    warnings.filterwarnings("ignore")
    plt.ion()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    # if given, load a pretrained model. .pth file must be the entire model, not just a state_dict
    if args.model is not None:
        model = torch.load(args.model)
    else:
        model = get_model(args.frozen)
    model = model.to(device)

    train_aug = 'trainaug' + args.aug
    val_aug = 'valaug1' if args.aug == 1 else 'valaug2'

    data_loaders, dataset_sizes, class_names = load_data(args.directory)

    with open("params.json", "r") as params_file:
        params = json.load(params_file)

    # training and validation phases
    model = train_model(model)
    torch.save(model, 'trained_model.pth')

    # show some results of the training
    get_metrics()

    exit(0)
