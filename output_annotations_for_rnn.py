import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
from collections import defaultdict
import json

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import model
import pdb


from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer


assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--model', help='Path to model (.pt) file.')

    parser = parser.parse_args(args)

    if parser.dataset == 'coco':
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'csv':
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]), is_visualizing=True)
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    retinanet.load_state_dict(torch.load(parser.model))

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.eval()

    scores_for_rnn = {}

    for idx, data in enumerate(dataloader_val):
        print(idx)

        with torch.no_grad():
            img_name = data['img_name'][0]
            scale = data['scale'][0]
            scores, transformed_anchors = retinanet(data['img'].cuda().float(), return_all_scores=True)
            transformed_anchors /= scale
            scores, transformed_anchors = scores.cpu(), transformed_anchors.cpu()
            scores = [[scores[i,j].item() for j in range(scores.size(1))] for i in range(scores.size(0))]
            transformed_anchors = [[transformed_anchors[i,j].item() for j in range(transformed_anchors.size(1))] for i in range(transformed_anchors.size(0))]
            curr = {'scores': scores, 'bboxes': transformed_anchors}
            scores_for_rnn[img_name] = curr

    with open('detections.json', 'w') as f:
        json.dump(scores_for_rnn, f)


if __name__ == '__main__':
 main()