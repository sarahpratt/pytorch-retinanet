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
import csv_eval

assert torch.__version__.split('.')[1] == '4'

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

	parser = parser.parse_args(args)

	log_dir = "./runs/" + parser.title
	writer = SummaryWriter(log_dir)

	#pdb.set_trace()

	with open(log_dir + '/config.csv', 'w') as f:
		for item in vars(parser):
			print(item + ',' + str(getattr(parser, item)))
			f.write(item + ',' + str(getattr(parser, item)) + '\n')

	if not os.path.isdir(log_dir + "/checkpoints"):
		os.makedirs(log_dir + "/checkpoints")

	if not os.path.isdir(log_dir + '/map_files'):
		os.makedirs(log_dir + '/map_files')

	if parser.csv_train is None:
		raise ValueError('Must provide --train-file when training,')

	if parser.csv_classes is None:
		raise ValueError('Must provide --classes-file when training')

	dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

	if parser.csv_val is None:
		dataset_val = None
		print('No validation annotations provided.')
	else:
		dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

	sampler = AspectRatioBasedSampler(dataset_train, batch_size=8, drop_last=False)
	dataloader_train = DataLoader(dataset_train, num_workers=8, collate_fn=collater, batch_sampler=sampler)

	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
		dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=sampler_val)

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

	if parser.resume_model:
		x = torch.load(parser.resume_model)
		if parser.reinit_classifier:
			dummy = nn.Conv2d(256, 9*dataset_train.num_classes(), kernel_size=3, padding=1)
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
	#torch.nn.DataParallel(retinanet).cuda()

	retinanet.training = True

	optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	loss_hist = collections.deque(maxlen=500)

	retinanet.train()
	retinanet.module.freeze_bn()

	print('Num training images: {}'.format(len(dataset_train)))

	for epoch_num in range(parser.resume_epoch, parser.epochs):

		retinanet.train()
		retinanet.module.freeze_bn()
		
		epoch_loss = []
		i = 0
		avg_class_loss = 0.0
		avg_reg_loss = 0.0

		for iter_num, data in enumerate(dataloader_train):
			i += 1

			try:
				optimizer.zero_grad()

				#pdb.set_trace()

				shape = data['img'].shape[2] * data['img'].shape[3]
				writer.add_scalar("train/image_shape", shape, epoch_num * (len(dataloader_train)) + i)

				classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda().float()])

				classification_loss = classification_loss.mean()
				regression_loss = regression_loss.mean()

				avg_class_loss += classification_loss
				avg_reg_loss += regression_loss

				if i % 100 == 0:
					writer.add_scalar("train/classification_loss", avg_class_loss / 100,
									  epoch_num * (len(dataloader_train)) + i)
					writer.add_scalar("train/regression_loss", avg_reg_loss / 100,
									  epoch_num * (len(dataloader_train)) + i)
					avg_class_loss = 0.0
					avg_reg_loss = 0.0

				loss = classification_loss + regression_loss

				if bool(loss == 0):
					continue

				loss.backward()

				torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

				optimizer.step()

				loss_hist.append(float(loss))

				epoch_loss.append(float(loss))

				print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

				del classification_loss
				del regression_loss
			except Exception as e:
				print(e)
				continue

		if epoch_num%2 == 0:

			print('Evaluating dataset')

			retinanet.eval()
			mAP, AP_string = csv_eval.evaluate(dataset_val, retinanet.module, score_threshold=0.1)
			with open(log_dir + '/map_files/{}_retinanet_{}.txt'.format(parser.dataset, epoch_num), 'w') as f:
				f.write(AP_string)
			total = 0.0
			all = 0.0
			total_unweighted = 0.0
			for c in mAP:
				total += mAP[c][0]*mAP[c][1]
				total_unweighted += mAP[c][0]
				all += mAP[c][1]
			writer.add_scalar("val/mAP", total/all, epoch_num)
			writer.add_scalar("val/mAP_unweighted", total_unweighted / len(mAP), epoch_num)


		scheduler.step(np.mean(epoch_loss))


		torch.save(retinanet.module.state_dict(), log_dir + '/checkpoints/{}_retinanet_{}.pth'.format(parser.dataset, epoch_num))

	retinanet.eval()

	torch.save(retinanet.module.state_dict(), log_dir + '/checkpoints/model_final.pth'.format(epoch_num))

if __name__ == '__main__':
 main()
