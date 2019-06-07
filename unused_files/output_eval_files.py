import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

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
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    #retinanet = torch.load(parser.model)
    retinanet = model.resnet50(num_classes=80, pretrained=True)
    retinanet.load_state_dict(torch.load(parser.model))

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.eval()

    if not os.path.isdir("./detection_files"):
        os.makedirs("./detection_files")

    for idx, data in enumerate(dataloader_val):

        with torch.no_grad():
            st = time.time()
            scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            print('Elapsed time: {}'.format(time.time()-st))
            idxs = np.where(scores>0.35)
            img_name = data['img_name'].split('.')[0]
            with open("./detection_files/" + img_name + '.txt', 'w') as f:
                for j in range(idxs[0].shape[0]):
                    bbox = transformed_anchors[idxs[0][j], :]
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                    f.write('{},{},{},{},label_name'.format(x1, y1, x2, y2, label_name))
                    if j < idxs[0].shape[0]-1:
                        f.write('\n')

if __name__ == '__main__':
 main()