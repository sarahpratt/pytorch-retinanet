import os
import argparse
import pdb
import json
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter

import math
import model
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

from imsitu_eval import BboxEval

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

	dataset_train = CSVDataset(train_file=parser.train_file, class_list=parser.classes_file, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

	if parser.val_file is None:
		dataset_val = None
		print('No validation annotations provided.')
	else:
		dataset_val = CSVDataset(train_file=parser.val_file, class_list=parser.classes_file, transform=transforms.Compose([Normalizer(), Resizer()]))

	sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=True)
	dataloader_train = DataLoader(dataset_train, num_workers=64, collate_fn=collater, batch_sampler=sampler)

	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=parser.batch_size, drop_last=True)
		dataloader_val = DataLoader(dataset_val, num_workers=64, collate_fn=collater, batch_sampler=sampler_val)

	print("loading dev")
	with open('./dev.json') as f:
		dev_gt = json.load(f)
	print("loading imsitu_dpace")
	with open('./imsitu_space.json') as f:
		all = json.load(f)
		verb_orders = all['verbs']

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

	# use_gpu = True

	# if use_gpu:
	# 	retinanet = retinanet.cuda()

	retinanet = torch.nn.DataParallel(retinanet).cuda()
	optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	if parser.all_box_regression:
		retinanet.all_box_regression = True

	print('Num training images: {}'.format(len(dataset_train)))

	x = torch.load('./runs/verb_in_lstm/checkpoints/retinanet_23.pth')
	retinanet.module.load_state_dict(x)
	# parser.resume_epoch = 1

	for epoch_num in range(parser.resume_epoch, parser.epochs):

		retinanet.train()
		retinanet.module.freeze_bn()
		
		epoch_loss = []
		i = 0
		avg_class_loss = 0.0
		avg_reg_loss = 0.0
		avg_bbox_loss = 0.0
		avg_verb_loss = 0.0
		retinanet.training = True

		now = time.time()

		all_data_time = 0.0
		all_model_time = 0.0
		all_backward_time = 0.0
		all_time = 0.0

		for iter_num, data in enumerate(dataloader_train):
			i += 1

			optimizer.zero_grad()
			image = data['img'].cuda().float()
			annotations = data['annot'].cuda().float()
			verbs = data['verb_idx'].cuda()
			widths = data['widths'].cuda()
			heights = data['heights'].cuda()

			all_data_time += time.time() - now
			now1 = time.time()

			class_loss, reg_loss, verb_loss, bbox_loss = retinanet([image, annotations, verbs, widths, heights])

			all_model_time += time.time() - now1
			now1 = time.time()

			avg_class_loss += class_loss.mean().item()
			avg_reg_loss += reg_loss.mean().item()
			avg_bbox_loss += bbox_loss.mean().item()
			avg_verb_loss += verb_loss.mean().item()


			if i % 100 == 0:
				print(
					'Epoch: {} | Iteration: {} | Class loss: {:1.5f} | Reg loss: {:1.5f} | Verb loss: {:1.5f} | Box loss: {:1.5f}'.format(
						epoch_num, iter_num, float(avg_class_loss/100), float(avg_reg_loss/100),
						float(avg_verb_loss/100), float(avg_bbox_loss/100)))
				writer.add_scalar("train/classification_loss", avg_class_loss / 100,
								  epoch_num * len(dataloader_train) + i)
				writer.add_scalar("train/regression_loss", avg_reg_loss / 100,
								  epoch_num * (len(dataloader_train)) + i)
				writer.add_scalar("train/bbox_loss", avg_bbox_loss / 100,
								  epoch_num * (len(dataloader_train)) + i)
				writer.add_scalar("train/verb_loss", avg_verb_loss / 100,
								  epoch_num * (len(dataloader_train)) + i)

				avg_class_loss = 0.0
				avg_reg_loss = 0.0
				avg_bbox_loss = 0.0
				avg_verb_loss = 0.0
				# print(time.time() - now)
				# now = time.time()

				# print("data")
				# print(all_data_time/10.0)
				# print("model")
				# print(all_model_time / 10.0)
				# print("backward")
				# print(all_backward_time / 10.0)
				# print("all")
				# print(all_time / 10.0)

				all_data_time = 0.0
				all_model_time = 0.0
				all_backward_time = 0.0
				all_time = 0.0

			loss = class_loss.mean() + reg_loss.mean() + bbox_loss.mean() + verb_loss.mean()
			#loss = verb_loss.mean()

			#epoch_loss.append(loss)

			if bool(loss == 0):
				continue
			loss.backward()
			torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 1, norm_type="inf")
			optimizer.step()

			all_backward_time +=  time.time() - now1

			all_time += time.time() - now

			now = time.time()


		torch.save(retinanet.module.state_dict(), log_dir + '/checkpoints/retinanet_{}.pth'.format(epoch_num))

		if epoch_num%1 == 0:
			evaluator = BboxEval()
			print('Evaluating dataset')
			retinanet.training = False
			retinanet.eval()
			k = 0
			for iter_num, data in enumerate(dataloader_val):

				if k%100 == 0:
					print(str(k) + " out of " + str(len(dataset_val)/parser.batch_size))
				k += 1
				x = data['img'].cuda().float()
				y = data['verb_idx'].cuda()
				widths = data['widths'].cuda()
				heights = data['heights'].cuda()

				verb_guess, noun_predicts, bbox_predicts, bbox_exists = retinanet([x, y, widths, heights])
				for i in range(len(verb_guess)):
					image = data['img_name'][i].split('/')[-1]
					verb = dataset_train.idx_to_verb[verb_guess[i]]
					nouns = []
					bboxes = []
					for j in range(6):
						if dataset_train.idx_to_class[noun_predicts[j][i]] == 'not':
							nouns.append('')
						else:
							nouns.append(dataset_train.idx_to_class[noun_predicts[j][i]])
						if bbox_exists[j][i] > 0.5:
							bboxes.append(bbox_predicts[j][i] / data['scale'][i])
						else:
							bboxes.append(None)
					verb_gt, nouns_gt, boxes_gt = get_ground_truth(image, dev_gt[image], verb_orders)
					evaluator.update(verb, nouns, bboxes, verb_gt, nouns_gt, boxes_gt, verb_orders)

			writer.add_scalar("val/verb_acc", evaluator.verb(), epoch_num)
			writer.add_scalar("val/value", evaluator.value(), epoch_num)
			writer.add_scalar("val/value_all", evaluator.value_all(), epoch_num)
			writer.add_scalar("val/value_bbox", evaluator.value_bbox(), epoch_num)
			writer.add_scalar("val/value_all_bbox", evaluator.value_all_bbox(), epoch_num)

		#epoch_loss = torch.Tensor(epoch_loss)
		#scheduler.step(epoch_loss.mean())
	retinanet.eval()
	torch.save(retinanet.module.state_dict(), log_dir + '/checkpoints/model_final.pth'.format(epoch_num))

def get_ground_truth(image, image_info, verb_orders):
	verb = image.split("_")[0]
	nouns = []
	bboxes = []
	for role in verb_orders[verb]["order"]:
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
