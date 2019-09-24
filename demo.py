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

    nouns_set = set()
    for noun in dataset_val.classes:
        if noun != "oov":
            nouns_set.add(noun.split('_')[0])


    return dataset_val, dev_gt, verb_orders, all_idx_to_english, nouns_set


def load_model(dataset_val):
    retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(retinanet).cuda()

    x = torch.load('./retinanet_30.pth')
    retinanet.module.load_state_dict(x['state_dict'])

    return retinanet


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

def get_color(evaluator, boxes, gt_box, nouns, gt_nouns):
    print(gt_box)
    if gt_box != None:
        gt_box = [item * 2 for item in gt_box]
    if nouns in gt_nouns and (evaluator.bb_intersection_over_union(boxes, gt_box)):
        return (0, 255, 0)
    return (255, 15, 119)


@st.cache
def categorize_ims_by_verb(dataset_val):
    verb_categorizations = defaultdict(list)
    i = 0
    for image_name in dataset_val.image_names:
        image = image_name.split('/')[-1].split('_')[0]
        verb_categorizations[image].append(i)
        i += 1
    return verb_categorizations


@st.cache
def load_baselines():
    with open('./baseline_files/gt_baseline_fixed.json') as f:
        gt = json.load(f)
    with open('./baseline_files/top1_baseline_fixed.json') as f:
        top1 = json.load(f)
    return gt, top1


evaluator = BboxEval()
dataset_val, dev_gt, verb_orders, all_idx_to_english, nouns_set = load_jsons()
retinanet = load_model(dataset_val)
retinanet.training = False
retinanet.eval()
verb_categorizations = categorize_ims_by_verb(dataset_val)
gt_baseline, top1_baseline = load_baselines()


retinanet.training = False

option_list = []
for key in verb_categorizations:
    option_list.append(key)


with open('verbs.txt', 'w') as f:
    option_list = ['random'] + sorted(option_list)
    for option in option_list:
        f.write(option)
        f.write('\n')

selected_verb = st.selectbox('Pick a verb or select random to generate a random image', option_list, index=0)
use_gt_verb = st.radio('Predict Verb or use GT', ["GT Verb", "Predict Verb"],  index=0)
print()
st.button('generate')

v = selected_verb
if v == 'random':
    data_idx = random.randint(0, dataset_val.__len__())
else:
    data_idx = random.choice(verb_categorizations[v])
data = dataset_val.__getitem__(data_idx)


im_data = data['img'].cuda().float().unsqueeze(0).permute(0, 3, 1, 2)
verb_idx_data = torch.tensor(data['verb_idx']).unsqueeze(0)
widths = torch.tensor(im_data.shape[2]).unsqueeze(0).cuda().float()
heights = torch.tensor(im_data.shape[3]).unsqueeze(0).cuda().float()

image = data['img_name'].split('/')[-1]

gt_value = use_gt_verb == "GT Verb"
print(use_gt_verb)
#retinanet.use_gt_verb = True

verb_guess, noun_predicts, bbox_predicts, bbox_exists = retinanet([im_data, verb_idx_data, widths, heights], widths, False, False, gt_value)

if gt_value:
    baseline = gt_baseline[image]
else:
    baseline = top1_baseline[image]

baseline_verb = baseline['verb']
baseline_noun = baseline['nouns']
baseline_boxes = baseline['boxes']

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
order_baseline = verb_orders[baseline_verb]["order"]
order_pred = verb_orders[verb]["order"]
st.markdown("<center><pre>Ground Truth: " + verb_gt + "            Baseline: " + baseline_verb + "            Model: " + verb + "   " + "</pre></center>")

im_size = 500

for i in range(6):

    # if len(order_gt) <= i and len(order_pred) <= i:
    #     break

    if len(order_gt) <= i:
        im_file_gt = Image.new('RGB', (im_size, im_size), "white")
        cap_gt = " "
    else:
        color = (0, 0, 255)
        cap_gt = order_gt[i] + ": "
        for noun in nouns_gt[i]:
            if noun == '' or noun in nouns_set:
                color = (255, 255, 255)
            if noun == '':
                w = 'N/A'
            elif noun in all_idx_to_english:
                w = all_idx_to_english[noun]['gloss'][0]
            else:
                w = noun
            cap_gt += w + ", "

        im_file_gt = evaluator.visualize_for_demo(boxes_gt[i], image, color)



    if len(order_baseline) <= i:
        im_file_baseline = Image.new('RGB', (im_size, im_size), "white")
        cap_baseline = " "
    else:
        print(baseline_boxes[i])
        print(image)
        if baseline_boxes[i] == 'None':
            bbox = None
        else:
            bbox = [item*2 for item in baseline_boxes[i]]
        color = get_color(evaluator, bbox, boxes_gt[i], baseline_noun[i], nouns_gt[i])
        im_file_baseline = evaluator.visualize_for_demo(bbox, image, color)
        if baseline_noun[i] == '':
            w = 'N/A'
        elif baseline_noun[i] in all_idx_to_english:
            w = all_idx_to_english[baseline_noun[i]]['gloss'][0]
        else:
            w = baseline_noun[i]
        cap_baseline = order_baseline[i] + ": " + w


    if len(order_pred) <= i:
        im_file = Image.new('RGB', (im_size, im_size), "white")
        cap_pred = " "
    else:
        color = get_color(evaluator, bboxes[i], boxes_gt[i], nouns[i], nouns_gt[i])
        im_file = evaluator.visualize_for_demo(bboxes[i], image, color)
        if nouns[i] == '':
            w = 'N/A'
        elif nouns[i] in all_idx_to_english:
            w = all_idx_to_english[nouns[i]]['gloss'][0]
        else:
            w = nouns[i]
        cap_pred = order_pred[i] + ": " + w


    im_file_gt.thumbnail((im_size, im_size), Image.ANTIALIAS)
    im_file_baseline.thumbnail((im_size, im_size), Image.ANTIALIAS)
    im_file.thumbnail((im_size, im_size), Image.ANTIALIAS)

    width = im_file_gt.size[0]
    h = im_file_gt.size[1]
    new_im_gt = Image.new("RGB", (im_size, im_size), 'white')
    new_im_gt.paste(im_file_gt, (int((im_size - width)/2), int(im_size - h)))

    width = im_file_baseline.size[0]
    h = im_file_baseline.size[1]
    new_im_gt_baseline = Image.new("RGB", (im_size, im_size), 'white')
    new_im_gt_baseline.paste(im_file_baseline, (int((im_size - width) / 2), int(im_size - h)))

    width = im_file.size[0]
    h = im_file.size[1]
    new_im_gt_pred = Image.new("RGB", (im_size, im_size), 'white')
    new_im_gt_pred.paste(im_file, (int((im_size - width)/2), int(im_size - h)))

    if len(cap_gt) > 20:
        cap_gt = cap_gt[:20]

    if len(cap_gt) > 70:
        cap_gt = cap_gt[:70]


    st.image([new_im_gt, new_im_gt_baseline, new_im_gt_pred], caption=[cap_gt, cap_baseline, cap_pred])
    #st.image([new_im_gt], caption=[cap_gt])



