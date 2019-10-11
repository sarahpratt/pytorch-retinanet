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

import math
import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import json
import csv_eval

#assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):

	parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

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

	#pdb.set_trace()

	print("loading dev")
	with open('./dev.json') as f:
		dev_gt = json.load(f)

	with open(log_dir + '/config.csv', 'w') as f:
		for item in vars(parser):
			print(item + ',' + str(getattr(parser, item)))
			f.write(item + ',' + str(getattr(parser, item)) + '\n')

	if not os.path.isdir(log_dir + "/checkpoints"):
		os.makedirs(log_dir + "/checkpoints")

	if not os.path.isdir(log_dir + '/map_files'):
		os.makedirs(log_dir + '/map_files')

	dataset_train = CSVDataset(train_file=parser.train_file, class_list=parser.classes_file,
							   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

	if parser.val_file is None:
		dataset_val = None
		print('No validation annotations provided.')
	else:
		dataset_val = CSVDataset(train_file=parser.val_file, class_list=parser.classes_file,
								 transform=transforms.Compose([Normalizer(), Resizer()]))

	sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=True)
	dataloader_train = DataLoader(dataset_train, num_workers=64, collate_fn=collater, batch_sampler=sampler)

	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=parser.batch_size, drop_last=False)
		dataloader_val = DataLoader(dataset_val, num_workers=64, collate_fn=collater, batch_sampler=sampler_val)

	print('creating model')
	# Create the model
	if parser.depth == 18:
		retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 34:
		retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 50:
		retinanet = model.resnet50(num_classes=dataset_train.num_classes(), num_nouns=dataset_train.num_nouns(), pretrained=True)
	elif parser.depth == 101:
		retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 152:
		retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
	else:
		raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

	print('done model')
	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()
	
	retinanet = torch.nn.DataParallel(retinanet).cuda()
	#torch.nn.DataParallel(retinanet).cuda()

	retinanet.training = True

	optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)

	retinanet.train()
	retinanet.module.freeze_bn()

	print('Num training images: {}'.format(len(dataset_train)))

	for epoch_num in range(parser.resume_epoch, parser.epochs):

		# train(retinanet, optimizer, dataloader_train, parser, epoch_num, writer)
		#
		# torch.save(retinanet.module.state_dict(), log_dir + '/checkpoints/retinanet_{}.pth'.format(epoch_num))

		eval(retinanet, dataloader_val, parser, dataset_val, dataset_train, dev_gt, epoch_num, writer)


def train(retinanet, optimizer, dataloader_train, parser, epoch_num, writer):
	retinanet.train()
	retinanet.module.freeze_bn()
	retinanet.training = True
	i = 0
	avg_class_loss = 0.0
	avg_reg_loss = 0.0
	avg_noun_loss = 0.0
	for iter_num, data in enumerate(dataloader_train):
		i += 1
		optimizer.zero_grad()

		classification_loss, regression_loss, noun_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda().float()])

		classification_loss = classification_loss.mean()
		regression_loss = regression_loss.mean()
		noun_loss = noun_loss.mean()

		avg_class_loss += classification_loss
		avg_reg_loss += regression_loss
		avg_noun_loss += noun_loss

		if i % 100 == 0:
			writer.add_scalar("train/classification_loss", avg_class_loss / 100,
							  epoch_num * (len(dataloader_train)) + i)
			writer.add_scalar("train/regression_loss", avg_reg_loss / 100,
							  epoch_num * (len(dataloader_train)) + i)
			writer.add_scalar("train/noun_loss", avg_noun_loss / 100,
							  epoch_num * (len(dataloader_train)) + i)
			print(
				'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | noun loss: {:1.5f}'.format(
					epoch_num, iter_num, float(avg_class_loss/100), float(avg_reg_loss/100), float(avg_noun_loss/100)))

			avg_class_loss = 0.0
			avg_reg_loss = 0.0
			avg_noun_loss = 0.0

		loss = classification_loss + regression_loss + noun_loss

		if bool(loss == 0):
			continue

		loss.backward()

		torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

		optimizer.step()


def eval(retinanet, dataloader_val, parser, dataset_val, dataset_train, dev_gt, epoch_num, writer):
	i = 0
	print('evaluating')
	retinanet.eval()
	retinanet.training = False
	total_boxes = 0.0
	total_with_overlap = 0.0
	total_with_overlap_20 = 0.0

	total_overlapping_gt = 0.0
	total_overlapping_gt_correct = 0.0

	for iter_num, data in enumerate(dataloader_val):
		i += 1
		gt = data['annot'].cuda()
		proposals, nouns = retinanet(data['img'].cuda().float())
		for batch_num in range(gt.shape[0]):
			im_anns = gt[batch_num]
			y = calc_iou(proposals[batch_num], gt[batch_num, :, :4])
			for box_num in range(gt.shape[1]):
				box = im_anns[box_num][:4]
				if box[0] != -1:
					total_boxes += 1
					max_iou = max(y[:, box_num])
					max_iou_20 =  max(y[:20, box_num])
					if max_iou > 0.5:
						total_with_overlap += 1
					if max_iou_20 > 0.5:
						total_with_overlap_20 += 1
			for all_boxes in range(100):
				box_dist = y[all_boxes]
				maximum, ind = torch.topk(box_dist, 1)
				if maximum > 0.5:
					total_overlapping_gt += 1
					correct = 0
					if nouns[all_boxes][batch_num] == im_anns[ind, -1].long():
						correct = 1
					if nouns[all_boxes][batch_num] == im_anns[ind, -2].long():
						correct = 1
					if nouns[all_boxes][batch_num] == im_anns[ind, -2].long():
						correct = 1

					total_overlapping_gt_correct += correct



	writer.add_scalar("val/top_100", total_with_overlap /total_boxes, epoch_num)
	writer.add_scalar("val/top_20", total_with_overlap_20 /total_boxes, epoch_num)
	writer.add_scalar("val/correct_classification", total_overlapping_gt_correct / total_overlapping_gt, epoch_num)



def calc_iou(a, b):
	area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

	iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1),
																		 b[:, 0])
	ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1),
																		 b[:, 1])

	iw = torch.clamp(iw, min=0)
	ih = torch.clamp(ih, min=0)

	ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

	ua = torch.clamp(ua, min=1e-8)

	intersection = iw * ih

	IoU = intersection / ua

	return IoU


if __name__ == '__main__':
 main()
