import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.utils.data import DataLoader
from datasets import SegmentationDataSet
from transformations import Compose, DenseTarget, RandomFlip, Resize_Sample
from transformations import MoveAxis, Normalize01, RandomCrop_USA, RandomCrop_John, ColorTransformations, ColorNoise
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from os import walk
import torch as t
import numpy as np
import torch.nn as nn


def get_files(path):
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        for names in filenames:
            files.append(dirpath + '/' + names)
    return files

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_model(device, cl):
    
    unet = smp.Unet('resnet152', classes=cl, activation=None, encoder_weights='imagenet')

    if t.cuda.is_available():
        unet.cuda()         
    
    unet = unet.to(device)
    return unet


def import_data_jem(args, batch_sz):
    inputs_train = get_files('./input_data/raw_john_handy/')
    inputs_validation = get_files('./input_data/raw_john_cam/')
    targets_train = get_files('./input_data/mask_john_handy/')
    targets_validation = get_files('./input_data/mask_john_cam/')
    
    transforms = Compose([
        DenseTarget(),
        MoveAxis(),
        Normalize01(),
        RandomCrop_John(),
        ])

    # train dataset
    dataset_train = SegmentationDataSet(inputs=inputs_train,
                                        targets=targets_train,
                                        transform=transforms)

    # validation dataset
    dataset_valid = SegmentationDataSet(inputs=inputs_validation,
                                        targets=targets_validation,
                                        transform=transforms)
    
    # train dataset
    dataset_sample = SegmentationDataSet(inputs=inputs_train,
                                        targets=targets_train,
                                        transform=transforms)

    # train dataloader
    dataloader_training = DataLoader(dataset=dataset_train,
                                    batch_size=batch_sz,
                                    shuffle=True
                                    )
    
    # train dataloader
    dataloader_sample = DataLoader(dataset=dataset_train,
                                    batch_size=batch_sz,
                                    shuffle=True
                                    )

    # validation dataloader
    dataloader_validation = DataLoader(dataset=dataset_valid,
                                    batch_size=batch_sz,
                                    shuffle=True)

    
    return dataloader_training, dataloader_validation, dataloader_sample


def import_data(args, batch_sz, set):
    if set == 'usa':
        inputs = get_files('./input_data/raw_usa/')
        targets = get_files('./input_data/mask_usa/')

    if set == 'john_handy':
        inputs = get_files('./input_data/raw_john_handy/')
        targets = get_files('./input_data/mask_john_handy/')

    if set == 'john_cam':
        inputs = get_files('./input_data/raw_john_cam/')
        targets = get_files('./input_data/mask_john_cam/')

    split = 0.8  

    inputs_train, inputs_valid = train_test_split(
        inputs,
        random_state=42,
        train_size=split,
        shuffle=True)

    targets_train, targets_valid = train_test_split(
        targets,
        random_state=42,
        train_size=split,
        shuffle=True)


    if set == 'usa':
        transforms = Compose([
        DenseTarget(),
        MoveAxis(),
        Normalize01(),
        RandomCrop_USA(),
        RandomFlip(),
        ])
    else:
        transforms = Compose([
        DenseTarget(),
        MoveAxis(),
        Normalize01(),
        RandomCrop_John(),
        RandomFlip(),
        ColorNoise()
        ])

    # train dataset
    dataset_train = SegmentationDataSet(inputs=inputs_train,
                                        targets=targets_train,
                                        transform=transforms)


    # validation dataset
    dataset_valid = SegmentationDataSet(inputs=inputs_valid,
                                        targets=targets_valid,
                                        transform=transforms)


    batchsize = batch_sz


    # train dataloader
    dataloader_training = DataLoader(dataset=dataset_train,
                                    batch_size=batchsize,
                                    shuffle=True
                                    )

    # validation dataloader
    dataloader_validation = DataLoader(dataset=dataset_valid,
                                    batch_size=batchsize,
                                    shuffle=True)

    
    
    return dataloader_training, dataloader_validation


def eval_classification(f, dload, device):
    corrects, losses = [], []
    for input, target in dload:
        input, target = input.to(device), target.to(device)
        logits = f(input)
        loss = nn.CrossEntropyLoss(reduce=False)(logits, target).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == target).float().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss


def checkpoint(f, tag, args, device, dload_train, dload_valid):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "train": dload_train,
        "valid": dload_valid
    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)


def logits2rgb(img):
    # Defined corporate design colors
    red = [200, 0, 10]
    green = [187,207, 74]
    blue = [0,108,132]
    yellow = [255,204,184]
    black = [0,0,0]
    white = [226,232,228]
    cyan = [174,214,220]
    orange = [232,167,53]

    colours = [red, green, blue, yellow, black, white, cyan, orange]

    shape = np.shape(img)
    h = int(shape[0])
    w = int(shape[1])
    col = np.zeros((h, w, 3))
    unique = np.unique(img)
    for i, val in enumerate(unique):
        mask = np.where(img == val)
        for j, row in enumerate(mask[0]):
            x = mask[0][j]
            y = mask[1][j]
            col[x, y, :] = colours[int(val)]

    return col.astype(int)

def mIOU(pred, label, num_classes=8):
    iou_list = list()
    present_iou_list = list()

    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).sum().item()
            union_now = pred_inds.sum().item() + target_inds.sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
        miou = np.mean(present_iou_list)
    return miou
