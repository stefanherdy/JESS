#!/usr/bin/env python3

import os
import pickle
import json
from utils import *
import torch as t
import torch.nn as nn 
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
import random

'''
Script Name: train_jess.py  
Author: Stefan Herdy  
Date: 15.06.2023  
Description:   
This is a the pytorch code implementation of Joint Energy-Based Semantic Image Segmentation. 
This script performs the JESS training. 
'''

# Fixed random seed
seed = 42 
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)

def main(args, test_number):
    print(args.test)
    print(args.energy)
    if args.test == 'jess':
        args.set = 'john'
        if args.energy == 'True':
            args.save_dir = './experiment/jess/True'
        else:
            args.save_dir = './experiment/jess/False'
        args.epochs = 100
    if args.test == 'norm':
        if args.set == 'usa':
            args.save_dir = './experiment/usa'
        if args.set == 'john_handy':
            args.save_dir = './experiment/john_handy'
        if args.set == 'john_cam':
            args.save_dir = './experiment/john_cam'
        args.epochs = 1000
    if args.test == 'norm':
        args.energy = 'False'

    makedirs(args.save_dir)
    with open(f'{args.save_dir}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)

    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    if args.test == 'jess' :
        dload_train, dload_valid, dload_sample = import_data_jem(args, args.batch_size)
    if args.test == 'norm':
        dload_train, dload_valid = import_data(args, args.batch_size, args.set)


    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    
    f = get_model(device, args.num_classes)
    
    params = f.parameters() 
    if args.optimizer == "adam":
        optim = t.optim.Adam(params, lr=args.learnrate, betas=[.9, .999], weight_decay=0.0)
    else:
        optim = t.optim.SGD(params, lr=args.learnrate, momentum=.9, weight_decay=0.0)

    best_valid_acc = 0.0
    iteration = 0

    train_losses = []
    val_losses = []
    val_corr = []
    
    for epoch in range(args.epochs):
        iter_losses = []
        for i, (x_val, _) in tqdm(enumerate(dload_valid)):
            x_val = x_val.to(device)
            x_train, y_train = next(iter(dload_train)) 
            x_train, y_train = x_train.to(device), y_train.to(device)
            Loss = 0.
            if args.energy == 'True': 
                x_adv, _ = next(iter(dload_sample))
                if t.cuda.is_available():
                    x_adv = x_adv.to(device)

                f_all = f(x_val)
                f_all = t.mean(f_all, (2,3))
                f_all = f_all.logsumexp(1)
                f_adv = f(x_adv)
                f_adv = t.mean(f_adv, (2,3))
                f_adv = f_adv.logsumexp(1)

                f_all = t.mean(f_all)
                f_adv = t.mean(f_adv)

                l_p_x = -(f_all - f_adv)
                Loss += args.p_x_weight * l_p_x

            logits = f(x_train)
            l_dis = nn.CrossEntropyLoss()(logits, y_train)
            Loss += l_dis
            iter_losses.append(Loss.item())

            optim.zero_grad()
            Loss.backward()
            optim.step()

        if args.energy == 'True':
            if iteration % args.print_every == 0:
                print('P(x) | {}:{:>d} f(x_val)={:>14.9f} f(x_adv)={:>14.9f} d={:>14.9f}'.format(epoch, i, f_all, f_adv,
                                                                                           f_all - f_adv))
        if iteration % args.print_every == 0:
            acc = (logits.max(1)[1] == y_train).float().mean()
            print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch,
                                                                            iteration,
                                                                            l_dis.item(),
                                                                            acc.item()))

        iteration += 1
        train_losses.append(np.mean(iter_losses))

        if epoch % args.eval_every == 0:
            f.eval()
            with t.no_grad():
                correct, loss = eval_classification(f, dload_valid, device)
                val_losses.append(loss)
                val_corr.append(correct)
                print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))
                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Best Valid!: {}".format(correct))
                    checkpoint(f, f'best_valid_ckpt_{test_number}.pt', args, device, dload_train, dload_valid)
            f.train()

        if epoch % args.ckpt_every == 0:
            checkpoint(f, f'ckpt_{epoch}_{test_number}.pt', args, device, dload_train, dload_valid)
            fig = plt.figure()
            plt.plot(train_losses)
            plt.plot(val_losses)
            plt.title('Loss')
            plt.legend(('Training', 'Validation'))
            plt.grid()
            
            makedirs("./records")
            with open(f"./records/accuracy_{args.energy}_{str(test_number)}_{str(args.set)}.txt" , "wb") as fp:   # Pickling Accuracy
                pickle.dump(val_corr, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("JESS")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size")
    parser.add_argument("--learnrate", type=int, default=0.0001, help='learn rate of optimizer')
    parser.add_argument("--p_x_weight", type=int, default=0.01, help='weight of energy based optimization')
    parser.add_argument("--optimizer", choices=['sgd', 'adam'], default='adam')
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=1, help="Epochs between print")
    parser.add_argument("--ckpt_every", type=int, default=20, help="Epochs between checkpoint save")
    parser.add_argument("--num_classes", type=int, default=8, help="Number of classes")
    parser.add_argument("--energy", choices=['True', 'False'], default='True', help="Set p(x) optimization on(True)/off(False)")
    parser.add_argument("--num_tests", type=int, default=6, help="Number of tests")
    parser.add_argument("--test", choices=['norm', 'jess'], default='norm', help="Normal test or Joint Energy-Based Sematic Segmentation")
    parser.add_argument("--set", choices=['usa', 'john_handy', 'john_cam'], default='usa', help="Dataset")
    args = parser.parse_args()

    for test_number in range(args.num_tests):
        main(args, test_number)

    visualize(args.num_tests)





    
    
