from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn

occ_counts = defaultdict(int)
multi = True

if multi:
    train_txt = '../annotations/train_anns_new.csv'
else:
    AP_txt = '../annotations/train_anns_new.csv'


if multi:
    AP_txt = './csv_retinanet_1_02_multi.txt'
else:
    AP_txt = './csv_retinanet_12_35_thresh.txt'


with open(multi) as f:
    for line in f:
        seen = set()
        category = line.split(',')[-1].split('\n')[0]
        occ_counts[category] += 1
        seen.add(category)
        if multi:
            category2 = line.split(',')[-2]
            if category2 != 'None' and category2 not in seen:
                occ_counts[category2] += 1
                seen.add(category2)
            category3 = line.split(',')[-3]
            if category3 != 'None' and category3 not in seen:
                occ_counts[category3] += 1


body_parts = ['n05564590_hand', 'n05563770_arm', 'n05216365_body', 'n05600637_face', 'n05538625_head', 'n05566504_finger', 'n05254795_hair', 'n05560787_leg', 'n05302499_mouth', 'n05305806_lip']
word_with_synonyms = ['n14844693_soil', 'n01320872_female', 'n09225146_body_of_water']


APs = []
occurances = []
colors = []
with open(AP_txt) as f:
    for line in f:
        category, AP = line.split(',')
        if occ_counts[category] < 5000:
            if category in body_parts:
                colors.append('green')
            elif category in word_with_synonyms:
                colors.append('blue')
            else:
                colors.append('red')
            APs.append(float(AP))
            occurances.append(occ_counts[category])

print(min(occurances))

seaborn.regplot(occurances, APs, n_boot=100, robust=True)
plt.scatter(occurances, APs, color=colors)
plt.show()




