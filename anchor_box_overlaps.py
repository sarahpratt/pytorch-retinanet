import torch
from torchvision import datasets, models, transforms
from anchors import Anchors
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader
import pdb



def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih
    IoU = intersection / ua
    return IoU


train_file = './annotations/top_80_lstm_small_ims/train_anns.csv'
classes_file = './annotations/top_80_lstm_small_ims/classes_80.csv'
crop_augment = False

dataset_train = CSVDataset(train_file=train_file, class_list=classes_file,
                           transform=transforms.Compose([Normalizer(), Augmenter(crop_augment), Resizer()]))

sampler = AspectRatioBasedSampler(dataset_train, batch_size=16, drop_last=True)
dataloader_train = DataLoader(dataset_train, num_workers=64, collate_fn=collater, batch_sampler=sampler)

anchors = Anchors()


total_without_overlap = 0.0
total_boxes = 0.0
sum_iou = 0.0
print(len(dataloader_train))
for iter_num, data in enumerate(dataloader_train):

    if iter_num%100 == 0:
        print(iter_num)
    image = data['img'].float()
    annotations = data['annot'].float()
    ancs = anchors(image)
    for i in range(6):
        IoU = calc_iou(ancs[0, :, :], annotations[:, i, :4])
        for ii in range(IoU.shape[1]):
            if annotations[ii, i, 0] != -1:
                maximum = max(IoU[:, ii])
                sum_iou += maximum
                total_boxes += 1.0
                if maximum < 0.5:
                    total_without_overlap += 1.0

pdb.set_trace()





