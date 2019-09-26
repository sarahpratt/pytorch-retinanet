from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import pdb

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

#from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage
import cv2
from PIL import Image


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, noun_list, transform=None, is_visualizing=False):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform
        self.is_visualizing = is_visualizing
        self.noun_list = noun_list

        # parse the provided class file
        try:
            with open(self.class_list, 'r') as file:
                self.classes, self.idx_to_role = self.load_classes(csv.reader(file, delimiter=','))
            with open(self.noun_list, 'r') as file:
                self.nouns, self.idx_to_noun = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            #raise_from(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)
            ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e))


        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with open(self.train_file, 'r') as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            print("Error")
            #raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
            ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e))

        self.image_names = list(self.image_data.keys())

        self.verb_to_idx = {}
        self.idx_to_verb = []
        with open('verb_indices.txt') as f:
            k = 0
            for line in f:
                verb = line.split('\n')[0]
                self.idx_to_verb.append(verb)
                self.verb_to_idx[verb] = k
                k += 1

        self.image_to_image_idx = {}
        i = 0
        for image_name in self.image_names:
            self.image_to_image_idx[image_name] = i
            i += 1



    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            #raise_from(ValueError(fmt.format(e)), None)
            ValueError(fmt.format(e))

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')


    def load_classes(self, csv_reader):
        result = {}
        id_to_name = []
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                #raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
                ValueError('line {}: format should be \'class_name,class_id\''.format(line))
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
            id_to_name.append(class_name)
        return result, id_to_name


    def __len__(self):
        return len(self.image_names)
        #return 100

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        verb = self.image_names[idx].split('/')[2]
        verb = verb.split('_')[0]
        verb_idx = self.verb_to_idx[verb]
        sample = {'img': img, 'annot': annot, 'img_name': self.image_names[idx], 'verb_idx': verb_idx}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        if img.shape[2] > 3:
            img = img[:, :, :3]

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 8))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            if a['x1'] == -1:
                x1 = a['x1']#SARAH - dividing these
                x2 = a['x2']
                y1 = a['y1']
                y2 = a['y2']
            else:
                x1 = a['x1'] / 2.0  # SARAH - dividing these
                x2 = a['x2'] / 2.0
                y1 = a['y1'] / 2.0
                y2 = a['y2'] / 2.0


            #if (x2-x1) < 1 or (y2-y1) < 1:
            #    continue

            #annotation = np.zeros((1, 5))
            annotation        = np.zeros((1, 8)) #allow for 3 annotations

            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4] = self.name_to_label(a['role_class'])
            annotation[0, 5]  = self.noun_to_label(a['class1'])
            annotation[0, 6] = self.noun_to_label(a['class2'])
            annotation[0, 7] = self.noun_to_label(a['class3'])
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, role_class, class1, class2, class3 = row[:9]
            except ValueError:
                print('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line))
                #raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)
                raise ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line))

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, role_class, class1, class2, class3) == ('', '', '', '', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1 and x1 != -1:
                print(row)
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1 and y1 != -1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if role_class not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class1, classes))

            #set class2 and class3 equal to class1 if they are none. Class1 is guarenteed a values and each unique class
            #is only counted once so this does not change the value
            if class2 == 'None':
                class2 = class1

            if class3 == 'None':
                class3 = class1

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'role_class': role_class, 'class1': class1, 'class2': class2, 'class3': class3})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def noun_to_label(self, name):
        return self.nouns[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def num_nouns(self):
        return max(self.nouns.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    img_names = [s['img_name'] for s in data]
    verb_indices = [s['verb_idx'] for s in data]
    verb_indices = torch.tensor(verb_indices)

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]

    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 8)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 8)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales, 'img_name': img_names, 'verb_idx': verb_indices, 'widths': torch.tensor(widths).float(), 'heights': torch.tensor(heights).float()}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=128, max_side=256):
        image, annots, image_name = sample['img'], sample['annot'], sample['img_name']

        rows_orig, cols_orig, cns_orig = image.shape

        smallest_side = min(rows_orig, cols_orig)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows_orig, cols_orig)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows_orig*scale)), int(round((cols_orig*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4][annots[:, :4] > 0] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale, 'img_name': image_name, 'verb_idx': sample['verb_idx']}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, augment_with_crops):
        self.augment_with_crops = augment_with_crops
        if augment_with_crops:
            print("augmentation working")

    def __call__(self, sample, flip_x=0.5):

        image, annots, img_name = sample['img'], sample['annot'], sample['img_name']


        if np.random.rand() < flip_x:
            #image, annots, img_name = sample['img'], sample['annot'], sample['img_name']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0][annots[:, 0] > 0] = cols - x2[annots[:, 0] > 0]
            annots[:, 2][annots[:, 2] > 0] = cols - x_tmp[annots[:, 2] > 0]

            sample = {'img': image, 'annot': annots, 'img_name': img_name, 'verb_idx': sample['verb_idx']}

        if self.augment_with_crops:
            crop_x_left = image.shape[1] * random.random() * .3
            crop_x_right = image.shape[1] * random.random() * .3
            x_1 = annots[:, 2] < crop_x_left
            x_2 = annots[:, 0] > (image.shape[1] - crop_x_right)

            annots[x_1 + x_2, :4] = -1

            x = annots[:, 0][annots[:, 0] != -1]
            y = annots[:, 2][annots[:, 2] != -1]
            annots[:, 0][annots[:, 0] != -1] = np.clip(x - crop_x_left, 0, image.shape[1] - crop_x_left - crop_x_right)
            annots[:, 2][annots[:, 2] != -1] = np.clip(y - crop_x_left, 0, image.shape[1] - crop_x_left - crop_x_right)

            crop_y_left = image.shape[0] * random.random() * .3
            crop_y_right = image.shape[0] * random.random() * .3

            x_1 = annots[:, 3] < crop_y_left
            x_2 = annots[:, 1] > (image.shape[0] - crop_y_right)

            annots[x_1 + x_2, :4] = -1

            x_second = annots[:, 1][annots[:, 1] != -1]
            y_second = annots[:, 3][annots[:, 3] != -1]
            annots[:, 1][annots[:, 1] != -1] = np.clip(x_second - crop_y_left, 0, image.shape[0] - crop_y_left - crop_y_right)
            annots[:, 3][annots[:, 3] != -1] = np.clip(y_second - crop_y_left, 0,  image.shape[0] - crop_y_left - crop_y_right)

            image = image[int(crop_y_left):int(image.shape[0]-crop_y_right), int(crop_x_left):int(image.shape[1]-crop_x_right), :]

            sample = {'img': image, 'annot': annots, 'img_name': img_name, 'verb_idx': sample['verb_idx']}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots, 'img_name': sample['img_name'], 'verb_idx': sample['verb_idx']}

        #return {'img': (image.astype(np.float32)), 'annot': annots, 'img_name': sample['img_name'], 'verb_idx': sample['verb_idx']}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
            #self.std = [1, 1, 1]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
