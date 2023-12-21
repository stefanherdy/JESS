#!/usr/bin/env python3

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch as t
import torch.nn as nn 
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from sklearn.metrics import precision_recall_fscore_support
import cv2
import segmentation_models_pytorch as smp
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True

'''
Script Name: evaluate_model.py  
Author: Stefan Herdy  
Date: 15.06.2023  
Description:   
This is a the pytorch code implementation of Joint Energy-Based Semantic Image Segmentation. 
This script evaluates the models trained in train_jess.py. 
'''

n_ch = 3
seed= 42 

os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)

def get_model_jess(device, i):
    unet_true = smp.Unet('resnet152', classes=args.num_classes, activation=None, encoder_weights='imagenet')
    if t.cuda.is_available():
        unet_true.cuda()     

    unet_false = smp.Unet('resnet152', classes=args.num_classes, activation=None, encoder_weights='imagenet')
    if t.cuda.is_available():
        unet_false.cuda()         
     
    print("Loading model")
    ckpt_true = t.load(f'./experiment/jess/True/best_valid_ckpt{i}.pt')
    unet_true.load_state_dict(ckpt_true["model_state_dict"])
    unet_true = unet_true.to(device)

    ckpt_false = t.load(f'./experiment/jess/False/best_valid_ckpt{i}.pt')
    unet_false.load_state_dict(ckpt_false["model_state_dict"])
    unet_false = unet_false.to(device)

    return unet_true, unet_false

def get_model(device):
    unet = smp.Unet('resnet152', classes=args.num_classes, activation=None, encoder_weights='imagenet')
    if t.cuda.is_available():
        unet.cuda()     

    print("Loading model")
    ckpt = t.load(f'./experiment/{args.set}/best_valid_ckpt.pt')
    unet.load_state_dict(ckpt["model_state_dict"])
    unet = unet.to(device)
    return unet


def predict_jess(args, model_true, model_false, dload, device):
    iou_list_true = []
    iou_list_false = []
    target_annotations_true = np.array([])
    predicted_annotations_true = np.array([])
    target_annotations_false = np.array([])
    predicted_annotations_false = np.array([])
    correct_list_true = []
    correct_list_false = []

    for i, (x_p_d, y_p_d) in enumerate(dload):
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        model_true.eval()
        logits_true = model_true(x_p_d)
        correct_true = np.mean((logits_true.max(1)[1] == y_p_d).float().cpu().numpy())
        logits_max_true = logits_true.max(1)[1].float().cpu().numpy()
        label = y_p_d.float().cpu().numpy()
        IOU = mIOU(logits_max_true, label)
        iou_list_true.append(IOU)
        correct_list_true.append(np.mean(correct_true))

        target_annotations_true= np.concatenate((target_annotations_true, (np.ndarray.flatten(np.array(label)))))
        predicted_annotations_true= np.concatenate((predicted_annotations_true, (np.ndarray.flatten(np.array(logits_max_true)))))
        model_false.eval()
        logits_false = model_false(x_p_d)
        correct_false = np.mean((logits_false.max(1)[1] == y_p_d).float().cpu().numpy())
        logits_max_false = logits_false.max(1)[1].float().cpu().numpy()
        IOU = mIOU(logits_max_false, label)
        iou_list_false.append(IOU)
        correct_list_false.append(np.mean(correct_false))

        target_annotations_false= np.concatenate((target_annotations_false, (np.ndarray.flatten(np.array(label)))))
        predicted_annotations_false= np.concatenate((predicted_annotations_false, (np.ndarray.flatten(np.array(logits_max_false)))))

    acc_true = np.mean(correct_list_true)
    acc_false = np.mean(correct_list_false)    
    iou_true = np.mean(iou_list_true)
    iou_false = np.mean(iou_list_false)
    pr_rec_f1_true = precision_recall_fscore_support(target_annotations_true, predicted_annotations_true, average='macro')
    pr_rec_f1_false = precision_recall_fscore_support(target_annotations_false, predicted_annotations_false, average='macro')

    return acc_true, acc_false, iou_true, iou_false, pr_rec_f1_true, pr_rec_f1_false

def predict(args, model, dload, device):
    iou_list = []
    correctlist = []
    target_annotations = np.array([])
    predicted_annotations = np.array([])
    for i, (x_p_d, y_p_d) in enumerate(dload):
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        model.eval()
        logits = model(x_p_d)
        correct = np.mean((logits.max(1)[1] == y_p_d).float().cpu().numpy())
        print('True: ' + str(i) + '_' + str(correct))
        correctlist.append(correct)        
        logits_max = logits.max(1)[1].float().cpu().numpy()
        label = y_p_d.float().cpu().numpy() 
        IOU = mIOU(logits_max, label)
        iou_list.append(IOU)
        print(IOU)
        target_annotations= np.concatenate((target_annotations, (np.ndarray.flatten(np.array(label)))))
        predicted_annotations= np.concatenate((predicted_annotations, (np.ndarray.flatten(np.array(logits_max)))))
    print('mIOU:')
    print(np.mean(iou_list))
    print('Accuracy:')
    print(np.mean(correctlist))
    print(precision_recall_fscore_support(target_annotations, predicted_annotations, average='macro'))

    iou = np.mean(iou_list)
    pr_rec_f1 = precision_recall_fscore_support(target_annotations, predicted_annotations, average='macro')
    accuracy = np.mean(correctlist)

    return accuracy, iou, pr_rec_f1

def evaluate(args):
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    print(args.test)
    if args.test == 'jess':
        dload_train, dload_valid, dload_sample = import_data_jem(args, args.batch_size)
    if args.test == 'norm':
        dload_train, dload_valid = import_data(args, args.batch_size, args.set)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    
    if args.test == 'jess':
        f_true, f_false = get_model_jess(device, i)
        with t.no_grad():
            acc_true, acc_false, iou_true, iou_false, pr_rec_f1_true, pr_rec_f1_false = predict_jess(args, f_true, f_false, dload_valid, device)
        return acc_true, acc_false, iou_true, iou_false, pr_rec_f1_true, pr_rec_f1_false

    if args.test == 'norm':
        f = get_model(device)
        with t.no_grad():
            predict(args, f, dload_valid, device)
        return accuracy, iou, pr_rec_f1
            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=['norm', 'jess'], default='norm', help="Normal test or Joint Energy-Based Sematic Segmentation")
    parser.add_argument("--set", choices=['usa', 'john_handy', 'john_cam'], default='usa', help="Dataset")
    parser.add_argument("--num_classes", type=int, default=8, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch Size")
    parser.add_argument("--num_tests", type=int, default=6, help="Number of tests")
    args = parser.parse_args()

    args.test = "jess"

    if args.test == 'jess':
        iou_true_list = []
        iou_false_list = []
        precision_true_list = []
        precision_false_list = []
        recall_true_list = []
        recall_false_list = []
        f1_true_list = []
        f1_false_list = []
        acc_true_list = []
        acc_false_list = []

        for i in range(args.num_tests):
            acc_true, acc_false, iou_true, iou_false, prf_true, prf_false = evaluate(args, i)

            iou_true_list.append(iou_true)
            iou_false_list.append(iou_false)
            precision_true_list.append(prf_true[0])
            recall_true_list.append(prf_true[1])
            f1_true_list.append(prf_true[2])
            precision_false_list.append(prf_false[0])
            recall_false_list.append(prf_false[1])
            f1_false_list.append(prf_false[2])
            acc_true_list.append(acc_true)
            acc_false_list.append(acc_false)

        print('IoU True:')
        print(np.mean(iou_true_list))
        print('IoU False:')
        print(np.mean(iou_false_list))
        print('Precision True:')
        print(np.mean(precision_true_list))
        print('Pecision False:')
        print(np.mean(precision_false_list))
        print('Recall True:')
        print(np.mean(recall_true_list))
        print('Recall False:')
        print(np.mean(recall_false_list))
        print('F1 True:')
        print(np.mean(f1_true_list))
        print('F1 False:')
        print(np.mean(f1_false_list))
        print('Accuracy True:')
        print(np.mean(acc_true_list))
        print('Accuracy False:')
        print(np.mean(acc_false_list))

    if args.test == "norm":
        iou_true_list = []
        precision_true_list = []
        recall_true_list = []
        f1_true_list = []
        for i in range(args.num_tests):
            acc, iou, prf = evaluate(args, i)

            iou_true_list.append(iou_true)
            precision_true_list.append(prf_true[0])
            f1_true_list.append(prf_true[2])
            recall_false_list.append(prf_false[1])

            accuracy, iou, pr_rec_f1 = evaluate(args, i)

        print('IoU True:')
        print(np.mean(iou_true_list))
        print('Precision True:')
        print(np.mean(precision_true_list))
        print('Recall True:')
        print(np.mean(recall_true_list))
        print('F1 True:')
        print(np.mean(f1_true_list))







    
    
