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

import warnings

from cosine_lr import CosineAnnealingWarmUpRestarts as CosScheduler

import math
import model
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

from imsitu_eval import BboxEval
from format_utils import cmd_to_title
import sys

#assert torch.__version__.split('.')[1] == '4'

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
	parser.add_argument("--detach_epoch", type=int, default=15)
	parser.add_argument("--gt_noun_epoch", type=int, default=9)
	parser.add_argument("--lr_decrease_epoch", type=int, default=100)
	parser.add_argument("--reinit-classifier", action="store_true", default=False)
	parser.add_argument("--rnn-weights", action="store_true", default=False)
	parser.add_argument("--augment", action="store_true", default=False)
	parser.add_argument("--init-with-verb-warmup", action="store_true", default=False)
	parser.add_argument("--cat-features", action="store_true", default=False)
	parser.add_argument("--rnn-class", action="store_true", default=False)
	parser.add_argument("--load-coco-weights", action="store_true", default=False)
	parser.add_argument("--just-verb-loss", action="store_true", default=False)
	parser.add_argument("--no-verb-loss", action="store_true", default=False)
	parser.add_argument("--just-class-loss", action="store_true", default=False)
	parser.add_argument("--retina-loss", action="store_true", default=False)
	parser.add_argument("--only-resnet", action="store_true", default=False)
	parser.add_argument("--warmup", action="store_true", default=False)
	parser.add_argument("--warmup2", action="store_true", default=False)
	parser.add_argument("--second-lr-decrease", type=int, default=100)
	parser.add_argument("--lr", type=float, default=.00001)
	parser.add_argument("--all-box-regression", action="store_true", default=False)
	parser.add_argument("--batch-size", type=int, default=16)
	parser = parser.parse_args(args)

	writer, log_dir = init_log_dir(parser)
	dataloader_train, dataset_train, dataloader_val, dataset_val = init_data(parser)

	print("loading dev")
	with open('./dev.json') as f:
		dev_gt = json.load(f)
	print("loading imsitu_dpace")
	with open('./imsitu_space.json') as f:
		all = json.load(f)
		verb_orders = all['verbs']

	role_tensor = get_roles_dictionary(verb_orders, dataset_train.verb_to_idx)

	warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead")

	print("loading model")
	retinanet = create_model(parser, dataset_train)
	retinanet.additional_class_branch = parser.rnn_class
	retinanet = torch.nn.DataParallel(retinanet).cuda()

	optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)
	#
	# if parser.only_resnet:
	# 	optimizer = optim.Adam(retinanet.module.feature_extractor.parameters(), lr=parser.lr)

	#optimizer = optim.SGD(retinanet.parameters(), lr=parser.lr, weight_decay=0.0001, momentum=0.9)
	#optimizer = optim.SGD(retinanet.parameters(), lr=parser.lr, weight_decay=0.0000305, momentum=0.875)

	#scheduler = CosScheduler(optimizer, T_0=30000, T_mult=1, eta_max=0.01, T_up=500, gamma=0.5)

	if parser.all_box_regression:
		retinanet.all_box_regression = True

	print('Num training images: {}'.format(len(dataset_train)))

	if parser.rnn_weights:
		load_rnn_weights(retinanet)

	if parser.load_coco_weights:
		load_coco_weights(retinanet)

	if parser.init_with_verb_warmup:
		x = torch.load('./verb_warm_up.pth')
		retinanet.module.load_state_dict(x)

	#load_old_weights(retinanet, './retinanet_50.pth')
	print("loading weights")
	x = torch.load('./retinanet_30.pth')
	retinanet.module.load_state_dict(x['state_dict'])
	# optimizer.load_state_dict(x['optimizer'])
	# for param_group in optimizer.param_groups:
	# 	param_group["lr"] = 0.00001

	print('weights loaded')

	#orig_resnet_weight = retinanet.module.layer4[2].conv3.weight
	#retinanet.module.feature_extractor.layer4.register_backward_hook(module_hook)

	for epoch_num in range(parser.resume_epoch, parser.epochs):

		#train(retinanet, optimizer, dataloader_train, parser, epoch_num, writer, role_tensor)

		#if epoch_num % 2 == 0:
		#	torch.save({'state_dict': retinanet.module.state_dict(), 'optimizer': optimizer.state_dict()}, log_dir + '/checkpoints/retinanet_{}.pth'.format(epoch_num))

		if epoch_num%1 == 0:
			print('Evaluating dataset')
			evaluate(retinanet, dataloader_val, parser, dataset_val, dataset_train, verb_orders, dev_gt, epoch_num,
					 writer, role_tensor)


	retinanet.eval()
	torch.save(retinanet.module.state_dict(), log_dir + '/checkpoints/model_final.pth'.format(epoch_num))


def module_hook(module, grad_input, grad_out):
	print("in")
	print(grad_input[0].abs().mean())
	print("out")
	print(grad_out[0].abs().mean())


def train(retinanet, optimizer, dataloader_train, parser, epoch_num, writer, role_tensor):
	retinanet.train()
	retinanet.module.freeze_bn()

	epoch_loss = []
	i = 0
	avg_class_loss = 0.0
	avg_reg_loss = 0.0
	avg_bbox_loss = 0.0
	avg_verb_loss = 0.0
	avg_rnn_class_loss = 0.0
	retinanet.training = True

	deatch_resnet = parser.detach_epoch > epoch_num
	use_gt_nouns = parser.gt_noun_epoch > epoch_num

	if epoch_num == parser.lr_decrease_epoch or epoch_num == parser.second_lr_decrease:
		lr = parser.lr / 10

		for param_group in optimizer.param_groups:
			param_group["lr"] = lr

	for iter_num, data in enumerate(dataloader_train):
		i += 1

		if parser.warmup and epoch_num == 0 and i <= 500:
			learning_rate = parser.lr/3.0 + (2.0*parser.lr/3.0)*(i/500.0)
			for param_group in optimizer.param_groups:
				param_group["lr"] = learning_rate


		if parser.warmup2 and epoch_num < 5:
			learning_rate = (parser.lr*(epoch_num * len(dataloader_train) + i))/(5 * len(dataloader_train))
			for param_group in optimizer.param_groups:
				param_group["lr"] = learning_rate


		optimizer.zero_grad()
		image = data['img'].cuda().float()
		annotations = data['annot'].cuda().float()
		verbs = data['verb_idx'].cuda()
		widths = data['widths'].cuda()
		heights = data['heights'].cuda()
		roles = role_tensor[verbs].cuda()

		class_loss, reg_loss, verb_loss, bbox_loss, all_rnn_class_loss = retinanet([image, annotations, verbs, widths, heights], roles,
															   deatch_resnet, use_gt_nouns)


		avg_class_loss += class_loss.mean().item()
		avg_reg_loss += reg_loss.mean().item()
		avg_bbox_loss += bbox_loss.mean().item()
		avg_verb_loss += verb_loss.mean().item()
		avg_rnn_class_loss += all_rnn_class_loss.mean().item()

		if i % 100 == 0:

			print(
				'Epoch: {} | Iteration: {} | Class loss: {:1.5f} | Reg loss: {:1.5f} | Verb loss: {:1.5f} | Box loss: {:1.5f}'.format(
					epoch_num, iter_num, float(avg_class_loss / 100), float(avg_reg_loss / 100),
					float(avg_verb_loss / 100), float(avg_bbox_loss / 100)))
			writer.add_scalar("train/classification_loss", avg_class_loss / 100,
							  epoch_num * len(dataloader_train) + i)
			writer.add_scalar("train/regression_loss", avg_reg_loss / 100,
							  epoch_num * (len(dataloader_train)) + i)
			writer.add_scalar("train/bbox_loss", avg_bbox_loss / 100,
							  epoch_num * (len(dataloader_train)) + i)
			writer.add_scalar("train/verb_loss", avg_verb_loss / 100,
							  epoch_num * (len(dataloader_train)) + i)
			writer.add_scalar("train/rnn_class_loss", avg_rnn_class_loss / 100,
							  epoch_num * (len(dataloader_train)) + i)

			avg_class_loss = 0.0
			avg_reg_loss = 0.0
			avg_bbox_loss = 0.0
			avg_verb_loss = 0.0
			avg_rnn_class_loss = 0.0

		if parser.just_verb_loss:
			loss = verb_loss.mean()
		elif parser.no_verb_loss:
			loss = class_loss.mean() + reg_loss.mean() + bbox_loss.mean() + all_rnn_class_loss.mean()
		elif parser.just_class_loss:
			loss = class_loss.mean()
		elif parser.retina_loss:
			loss = class_loss.mean() + reg_loss.mean()
		else:
			loss = class_loss.mean() + reg_loss.mean() + bbox_loss.mean() + verb_loss.mean() + all_rnn_class_loss.mean()

		if bool(loss == 0):
			continue
		loss.backward()
		torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 1, norm_type="inf")
		optimizer.step()
		#scheduler.step()


def evaluate(retinanet, dataloader_val, parser, dataset_val, dataset_train, verb_orders, dev_gt, epoch_num, writer, role_tensor):
	evaluator = BboxEval()
	retinanet.training = False
	retinanet.eval()
	k = 0
	for iter_num, data in enumerate(dataloader_val):

		if k % 100 == 0:
			print(str(k) + " out of " + str(len(dataset_val) / parser.batch_size))
		k += 1
		x = data['img'].cuda().float()
		y = data['verb_idx'].cuda()
		widths = data['widths'].cuda()
		heights = data['heights'].cuda()
		roles = role_tensor[y].cuda()

		verb_guess, noun_predicts, bbox_predicts, bbox_exists = retinanet([x, y, widths, heights], roles, use_gt_verb=False)
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
	writer.add_scalar("val/just_bbox", evaluator.just_bbox(), epoch_num)



def create_model(parser, dataset_train):
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
	return retinanet


def init_data(parser):
	dataset_train = CSVDataset(train_file=parser.train_file, class_list=parser.classes_file,
							   transform=transforms.Compose([Normalizer(), Augmenter(parser.augment), Resizer()]))

	if parser.val_file is None:
		dataset_val = None
		print('No validation annotations provided.')
	else:
		dataset_val = CSVDataset(train_file=parser.val_file, class_list=parser.classes_file,
								 transform=transforms.Compose([Normalizer(), Resizer()]))

	sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=True)
	dataloader_train = DataLoader(dataset_train, num_workers=64, collate_fn=collater, batch_sampler=sampler)

	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=parser.batch_size, drop_last=True)
		dataloader_val = DataLoader(dataset_val, num_workers=64, collate_fn=collater, batch_sampler=sampler_val)
	return dataloader_train, dataset_train, dataloader_val, dataset_val


def init_log_dir(parser):
	print()
	x = cmd_to_title(sys.argv[1:], True)
	print(x)
	log_dir = "./runs/" + x

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
	return writer, log_dir


def load_old_weights(retinanet, weights):
	x = torch.load(weights)
	just_resnet_weights = {}
	keys = retinanet.module.state_dict().keys()
	for weight in x:
		if weight not in keys:
			just_resnet_weights['feature_extractor.' + weight] = x[weight]
		else:
			just_resnet_weights[weight] = x[weight]
	model_dict = retinanet.module.state_dict()
	model_dict.update(just_resnet_weights)
	retinanet.module.load_state_dict(model_dict)


def load_rnn_weights(retinanet):
	x = torch.load('./best_39.pth.tar')
	just_resnet_weights = {}
	for weight in x['state_dict']:
		if 'feature_extractor' in weight and 'fc' not in weight:
			weight_in_model = weight.split('feature_extractor.')[-1]
			just_resnet_weights[weight_in_model] = x['state_dict'][weight]
	model_dict = retinanet.module.state_dict()
	model_dict.update(just_resnet_weights)
	retinanet.module.load_state_dict(model_dict)


def load_coco_weights(retinanet):
	x = torch.load('./coco_resnet_50_map_0_335_state_dict.pt')
	coco_weights = {}
	model_dict = retinanet.module.state_dict()
	for weight in x:
		if weight in model_dict:
			coco_weights[weight] = x[weight]
		#coco_weights['classificationModel.output_retina.weight'] = x['classificationModel.output.weight']
		#coco_weights['classificationModel.output_retina.bias'] = x['classificationModel.output.bias']
	model_dict.update(coco_weights)
	retinanet.module.load_state_dict(model_dict)


def get_roles_dictionary(verb_orders, verb_to_idx):
	role_dict = {}
	role_tensor = torch.zeros(504, 6)
	for verb in verb_orders:
		i = 0
		for role in verb_orders[verb]['order']:
			if role not in role_dict:
				role_dict[role] = len(role_dict.keys()) + 1
			role_tensor[verb_to_idx[verb], i] = role_dict[role]
			i += 1
	return role_tensor


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
