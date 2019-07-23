import time
import os
import copy
import argparse
import pdb
import collections
import sys
import json

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision
from tensorboardX import SummaryWriter

import math
import model
from anchors import Anchors
import losses
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

from imsitu_eval import BboxEval

import coco_eval
import csv_eval

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--train-file', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--classes-file', help='Path to file containing class list (see readme)')
    parser.add_argument('--val-file', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--title', type=str, default='')
    parser.add_argument("--resume_model", type=str, default="")
    parser.add_argument("--resume_epoch", type=int, default=0)
    parser.add_argument("--reinit-classifier", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=.00001)
    parser.add_argument("--all-box-regression", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=16)

    parser = parser.parse_args(args)

    log_dir = "./runs/" + parser.title
    writer = SummaryWriter(log_dir)

    with open(log_dir + '/config.csv', 'w') as f:
        for item in vars(parser):
            print(item + ',' + str(getattr(parser, item)))
            f.write(item + ',' + str(getattr(parser, item)) + '\n')

    if not os.path.isdir(log_dir + "/checkpoints"):
        os.makedirs(log_dir + "/checkpoints")
    if not os.path.isdir(log_dir + '/map_files'):
        os.makedirs(log_dir + '/map_files')
    if parser.train_file is None:
        raise ValueError('Must provide --train-file when training,')
    if parser.classes_file is None:
        raise ValueError('Must provide --classes-file when training')

    dataset_train = CSVDataset(train_file=parser.train_file, class_list=parser.classes_file,
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    if parser.val_file is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(train_file=parser.val_file, class_list=parser.classes_file,
                                 transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=True)
    dataloader_train = DataLoader(dataset_train, num_workers=8, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=parser.batch_size, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=8, collate_fn=collater, batch_sampler=sampler_val)

    print("loading dev")
    with open('./dev.json') as f:
        dev_gt = json.load(f)
    print("loading imsitu_dpace")
    with open('./imsitu_space.json') as f:
        all = json.load(f)
        verb_orders = all['verbs']
        all_idx_to_english = all['nouns']

    print("loading model")
    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    print("loading weights")
    if parser.resume_model:
        x = torch.load(parser.resume_model)
        if parser.reinit_classifier:
            dummy = nn.Conv2d(256, 9 * dataset_train.num_classes(), kernel_size=3, padding=1)
            x['classificationModel.output.weight'] = dummy.weight.clone()
            x['classificationModel.output.bias'] = dummy.bias.clone()
            prior = 0.01
            x['classificationModel.output.weight'].data.fill_(0)
            x['classificationModel.output.bias'].data.fill_(-math.log((1.0 - prior) / prior))
        retinanet.load_state_dict(x)

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet = torch.nn.DataParallel(retinanet).cuda()
    # torch.nn.DataParallel(retinanet).cuda()

    optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    if parser.all_box_regression:
        retinanet.all_box_regression = True

    print('Num training images: {}'.format(len(dataset_train)))

    x = torch.load('./retinanet_89.pth')
    retinanet.module.load_state_dict(x)

    evaluator = BboxEval()
    print('Evaluating dataset')
    retinanet.training = False
    retinanet.eval()
    k = 0
    for iter_num, data in enumerate(dataloader_val):
        if k % 100 == 0:
            print(str(k) + " out of " + str(len(dataset_val)))
        k += 1
        #pdb.set_trace()
        verb_guess, noun_predicts, bbox_predicts = retinanet([data['img'].cuda().float(), data['verb_idx']])
        #bbox_predicts /= data['scale']
        for i in range(len(verb_guess)):
            image = data['img_name'][i].split('/')[-1]
            verb = dataset_train.idx_to_verb[verb_guess[i]]
            nouns = []
            bboxes = []
            for j in range(2):
                #pdb.set_trace()
                if dataset_train.idx_to_class[noun_predicts[j][i]] == 'not':
                    nouns.append('')
                else:
                    nouns.append(dataset_train.idx_to_class[noun_predicts[j][i]])
                bboxes.append(bbox_predicts[j][i]/data['scale'][i])
            verb_gt, nouns_gt, boxes_gt = get_ground_truth(image, dev_gt[image], verb_orders)
            #scale = data['scale'][0]
            #pdb.set_trace()
            evaluator.update(verb, nouns, bboxes, verb_gt, nouns_gt, boxes_gt, verb_orders)
            image_file = evaluator.visualize(verb, nouns, bboxes, verb_gt, nouns_gt, boxes_gt, verb_orders, image, all_idx_to_english)
            image_file.save('./predictions/' + image)

    pdb.set_trace()
    writer.add_scalar("val/verb_acc", evaluator.verb(), 1)
    writer.add_scalar("val/value", evaluator.value(), 1)
    writer.add_scalar("val/value_all", evaluator.value_all(), 1)
    writer.add_scalar("val/value_bbox", evaluator.value_bbox(), 1)
    writer.add_scalar("val/value_all_bbox", evaluator.value_all_bbox(), 1)

    scheduler.step(np.mean(1))

    retinanet.eval()
    torch.save(retinanet.module.state_dict(), log_dir + '/checkpoints/model_final.pth'.format(1))


def get_ground_truth(image, image_info, verb_orders):
	verb = image.split("_")[0]
	nouns = []
	bboxes = []
	roles = ["agent", "tool"]
	#roles = ["agent"]
	if "agent" in verb_orders[verb]["order"] and "tool" in verb_orders[verb]["order"]:
		#for role in verb_orders[verb]["order"]:
		for role in roles:
			all_options = set()
			for i in range(3):
				all_options.add(image_info["frames"][i][role])
			nouns.append(all_options)
			if image_info["bb"][role][0] == -1:
				bboxes.append(None)
			else:
				b = [int(i) for i in image_info["bb"][role]]
				bboxes.append(b)
	return verb, nouns, bboxes


if __name__ == '__main__':
    main()
