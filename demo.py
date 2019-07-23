import streamlit as st
import json
from imsitu_eval import BboxEval
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
#matplotlib.use("agg")

import torch
from torchvision import datasets, models, transforms
import model
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer


val_file = './annotations/top_80_lstm/test_anns.csv'
classes_file = './annotations/top_80_lstm/classes_80.csv'


@st.cache
def load():
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

    x = torch.load('./retinanet_15.pth')
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


evaluator = BboxEval()
dataset_val, dev_gt, verb_orders, all_idx_to_english, retinanet = load()
retinanet.training = False

passage = st.text_area("passage", "running_16")
image_name = './images_512/' + str(passage) + '.jpg'
if image_name not in dataset_val.image_to_image_idx:
    st.write("That image does not exist in the val set")
else:
    data_idx = dataset_val.image_to_image_idx[image_name]
    data = dataset_val.__getitem__(data_idx)

    im_data = data['img'].cuda().float().unsqueeze(0).permute(0, 3, 1, 2)
    verb_idx_data = torch.tensor(data['verb_idx']).unsqueeze(0)
    verb_guess, noun_predicts, bbox_predicts, bbox_exists = retinanet([im_data, verb_idx_data])

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
    order = verb_orders[verb_gt]["order"]

    color = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)]

    #gt words
    im_file = evaluator.visualize_words_for_demo(order, verb_gt, nouns_gt, color, all_idx_to_english)
    st.image(im_file)

    #gt image
    im_file_gt = evaluator.visualize_for_demo(boxes_gt, image, color)
    st.image(im_file_gt)

    #pred words
    im_file = evaluator.visualize_words_for_demo_pred(order, verb, nouns, color, all_idx_to_english)
    st.image(im_file)

    #pred image
    im_file = evaluator.visualize_for_demo(bboxes, image, color)
    st.image(im_file)



#question = st.text_input("question", "Who stars in the Matrix?")
#st.write("dkfeef")


