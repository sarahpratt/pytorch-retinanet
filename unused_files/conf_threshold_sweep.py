import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision
from tensorboardX import SummaryWriter

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, \
    Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--title', type=str, default='')
    parser.add_argument("--resume_model", type=str, default="")
    parser.add_argument("--resume_epoch", type=int, default=0)

    parser = parser.parse_args(args)

    title = parser.resume_model.split('.')[0]

    log_dir = "./runs/" + title
    writer = SummaryWriter(log_dir)

    if not os.path.isdir(log_dir + "/checkpoints"):
        os.makedirs(log_dir + "/checkpoints")

    if not os.path.isdir(log_dir + '/map_files'):
        os.makedirs(log_dir + '/map_files')

    if parser.dataset == 'csv':

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_val.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_val.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_val.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_val.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if parser.resume_model:
        retinanet.load_state_dict(torch.load(parser.resume_model))

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    theshes = [.05, 0.1, 0.2, 0.3]

    i = 0
    for thresh in theshes:
        i = i + 1

        retinanet.eval()

        print('Evaluating dataset')

        mAP, AP_string = csv_eval.evaluate(dataset_val, retinanet, score_threshold=thresh)
        with open(log_dir + '/map_files/{}_retinanet_{}.txt'.format(parser.dataset, thresh), 'w') as f:
            f.write(AP_string)
        total = 0.0
        all = 0.0
        total_unweighted = 0.0
        for c in mAP:
            total += mAP[c][0] * mAP[c][1]
            total_unweighted += mAP[c][0]
            all += mAP[c][1]
        writer.add_scalar("thresh_finder/mAP", total / all, i)
        writer.add_scalar("thresh_finder/mAP_unweighted", total_unweighted / len(mAP), i)



if __name__ == '__main__':
    main()
