import streamlit as st
import json
from imsitu_eval import BboxEval
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
#matplotlib.use("agg")
import pdb
import pandas as pd
import numpy as np
from collections import defaultdict

import torch
from torchvision import datasets, models, transforms
import model
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from PIL import Image
import random

val_file = './annotations/top_80_lstm/test_anns.csv'
classes_file = './annotations/top_80_lstm/classes_80.csv'


@st.cache
def load_jsons():
    dataset_val = CSVDataset(train_file=val_file, class_list=classes_file, transform=transforms.Compose([Normalizer(), Resizer()]))
    print("loading dev")
    with open('./dev.json') as f:
        dev_gt = json.load(f)
    print("loading imsitu_dpace")
    with open('./imsitu_space.json') as f:
        all = json.load(f)
        verb_orders = all['verbs']
        all_idx_to_english = all['nouns']

    retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(retinanet).cuda()

    x = torch.load('./retinanet_4.pth')
    retinanet.module.load_state_dict(x)
    retinanet.training = False
    retinanet.eval()

    return dataset_val, dev_gt, verb_orders, all_idx_to_english, retinanet



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

@st.cache
def categorize_ims_by_verb(dataset_val):
    verb_categorizations = defaultdict(list)
    i = 0
    for image_name in dataset_val.image_names:
        image = image_name.split('/')[-1].split('_')[0]
        verb_categorizations[image].append(i)
        i += 1
    return verb_categorizations



evaluator = BboxEval()
dataset_val, dev_gt, verb_orders, all_idx_to_english, retinanet = load_jsons()
verb_categorizations = categorize_ims_by_verb(dataset_val)


option_list = []
for key in verb_categorizations:
    option_list.append(key)

option_list = ['random'] + sorted(option_list)
selected_verb = st.selectbox('Pick a verb or select random to generate a random image', option_list,  value=0)
st.button('generate')
v = option_list[selected_verb]
if v == 'random':
    data_idx = random.randint(0, dataset_val.__len__())
else:
    data_idx = random.choice(verb_categorizations[v])
data = dataset_val.__getitem__(data_idx)


im_data = data['img'].cuda().float().unsqueeze(0).permute(0, 3, 1, 2)
verb_idx_data = torch.tensor(data['verb_idx']).unsqueeze(0)
widths = torch.tensor(im_data.shape[2]).unsqueeze(0).cuda().float()
heights = torch.tensor(im_data.shape[3]).unsqueeze(0).cuda().float()

verb_guess, noun_predicts, bbox_predicts, bbox_exists = retinanet([im_data, verb_idx_data, widths, heights])

image = data['img_name'].split('/')[-1]
verb = dataset_val.idx_to_verb[verb_guess[0]]
nouns = []
bboxes = []
for j in range(6):
    if dataset_val.idx_to_class[noun_predicts[j][0]] == 'not':
        nouns.append('')
    else:
        nouns.append(dataset_val.idx_to_class[noun_predicts[j][0]])
    if bbox_exists[j][0] > 0.5:
        bboxes.append(bbox_predicts[j][0] / data['scale'])
    else:
        bboxes.append(None)

verb_gt, nouns_gt, boxes_gt = get_ground_truth(image, dev_gt[image], verb_orders)
evaluator.update(verb, nouns, bboxes, verb_gt, nouns_gt, boxes_gt, verb_orders)
order_gt = verb_orders[verb_gt]["order"]
order_pred = verb_orders[verb]["order"]
color = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)]
st.markdown("<center><pre><big> Ground Truth: " + verb_gt + "             Predicted: " + verb + "</big></pre></center>")

for i in range(6):

    if len(order_gt) <= i and len(order_pred) <= i:
        break

    if len(order_gt) <= i:
        im_file_gt = Image.new('RGB', (350, 350), "white")
        cap_gt = " "
    else:
        im_file_gt = evaluator.visualize_for_demo(boxes_gt[i], image, color[i])
        cap_gt = order_gt[i] + ": "
        for noun in nouns_gt[i]:
            if noun == '':
                w = 'not_applicable'
            elif noun in all_idx_to_english:
                w = all_idx_to_english[noun]['gloss'][0]
            else:
                w = noun
            cap_gt += w + ", "

    if len(order_pred) <= i:
        im_file = Image.new('RGB', (350, 350), "white")
        cap_pred = " "
    else:
        im_file = evaluator.visualize_for_demo(bboxes[i], image, color[i])
        if nouns[i] == '':
            w = 'not_applicable'
        elif nouns[i] in all_idx_to_english:
            w = all_idx_to_english[nouns[i]]['gloss'][0]
        else:
            w = nouns[i]
        cap_pred = order_pred[i] + ": " + w

    size = 350, 350

    im_file_gt.thumbnail(size, Image.ANTIALIAS)
    im_file.thumbnail(size, Image.ANTIALIAS)
    width = im_file_gt.size[0]
    h = im_file_gt.size[1]
    new_im_gt = Image.new("RGB", (350, 350), 'white')
    new_im_gt.paste(im_file_gt, ((350 - width)/2, 350 - h))

    width = im_file.size[0]
    h = im_file.size[1]
    new_im_gt_pred = Image.new("RGB", (350, 350), 'white')
    new_im_gt_pred.paste(im_file, ((350 - width)/2, 350 - h))

    st.image([new_im_gt, new_im_gt_pred], caption=[cap_gt, cap_pred])




