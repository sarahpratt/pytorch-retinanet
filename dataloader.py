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

from PIL import Image


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco      = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None, is_visualizing=False):
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

        # parse the provided class file
        try:
            with open(self.class_list, 'r') as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
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
        return result


    def __len__(self):
        return len(self.image_names)
        #return 100

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot, 'img_name': self.image_names[idx]}
        if self.is_visualizing:
            sample['is_visualizing'] = True
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
        annotations     = np.zeros((0, 7))

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

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            #annotation = np.zeros((1, 5))
            annotation        = np.zeros((1, 7)) #allow for 3 annotations
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class1'])
            annotation[0, 5] = self.name_to_label(a['class2'])
            annotation[0, 6] = self.name_to_label(a['class3'])
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class1, class2, class3 = row[:8]
            except ValueError:
                print('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line))
                #raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)
                raise ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line))

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class1, class2, class3) == ('', '', '', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                print(row)
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class1 not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class1, classes))
            if class2 not in classes and class2 != 'None':
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class2, classes))
            if class3 not in classes and class3 != 'None':
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class3, classes))

            #set class2 and class3 equal to class1 if they are none. Class1 is guarenteed a values and each unique class
            #is only counted once so this does not change the value
            if class2 == 'None':
                class2 = class1

            if class3 == 'None':
                class3 = class1

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class1': class1, 'class2': class2, 'class3': class3})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return 1

    def num_nouns(self):
        return len(self.classes)

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    img_names = [s['img_name'] for s in data]
        
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

        annot_padded = torch.ones((len(annots), max_num_annots, 7)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 7)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)
    if 'no_norm' in data[0]:
        no_norm_imgs = [s['no_norm'] for s in data]
        padded_imgs_no_norm = torch.zeros(batch_size, max_width, max_height, 3)
        for i in range(batch_size):
            no_norm_img = no_norm_imgs[i]
            padded_imgs_no_norm[i, :int(no_norm_img.shape[0]), :int(no_norm_img.shape[1]), :] = torch.Tensor(no_norm_img)

        return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales, 'img_name': img_names, 'no_norm': padded_imgs_no_norm}

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales, 'img_name': img_names}

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

        annots[:, :4] *= scale
        if 'no_norm' in sample:
            no_norm_image = sample['no_norm']
            no_norm_image = skimage.transform.resize(no_norm_image, (int(round(rows_orig * scale)), int(round((cols_orig * scale)))))
            new_image_no_norm = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
            new_image_no_norm[:rows, :cols, :] = no_norm_image.astype(np.float32)
            return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale, 'img_name': sample['img_name'], 'no_norm': new_image_no_norm}
        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale, 'img_name': image_name}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        image, annots, img_name = sample['img'], sample['annot'], sample['img_name']

        if np.random.rand() < flip_x:
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

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

        sample = {'img': image, 'annot': annots, 'img_name': img_name}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        if 'is_visualizing' in sample:
            return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots, 'img_name': sample['img_name'], 'no_norm': image.astype(np.float32)}
        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots, 'img_name': sample['img_name']}

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
