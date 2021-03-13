# Adapted from https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py

from __future__ import print_function

import math
import os.path
import warnings

import numpy as np
import scipy as sc
import torch.utils.data as data
from PIL import Image


def clamp(n, minn, maxn):
    return max(min(maxn, int(n)), minn)


def reshape_row(row):
    # scale
    std = 0.3
    row = np.array(list(map(lambda x: clamp(128 + (x / (2 * std)) * 128, 0, 255), row)))
    size = 32
    if len(row) > size**2:
        row = row[:size**2]
    number_of_rows = math.ceil(len(row) / size)
    padded_row = np.zeros(size * number_of_rows)
    padded_row[:len(row)] = row
    rows = padded_row.reshape(number_of_rows, size)
    result = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        j = int(number_of_rows * i / size)
        result[i, :] = rows[j]
    return result


class ActivityRecognitionDataset(data.Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        # now load the picked numpy arrays
        if self.split == 'train' or self.split == 'train_auto_quan' or self.split == 'val_auto_quan':
            self.train_data = self.getAllData("train")
            self.train_labels = self.getDataLabels("train") - 1
            reshaped_data = np.array(list(map(reshape_row, self.train_data)))
            self.train_data = np.array([reshaped_data, reshaped_data, reshaped_data])
            self.train_data = self.train_data.transpose((1, 2, 3, 0))  # convert to HWC
            self.train_labels = [int(x) for x in self.train_labels]
            pass
        else:
            self.test_data = self.getAllData("test")
            self.test_labels = self.getDataLabels("test") - 1
            transformed_data = np.array(list(map(reshape_row, self.test_data)))
            self.test_data = np.array([transformed_data, transformed_data, transformed_data])
            self.test_data = self.test_data.transpose((1, 2, 3, 0))  # convert to HWC
            self.test_labels = [int(x) for x in self.test_labels]
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'train' or self.split == 'train_auto_quan' or self.split == 'val_auto_quan':
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.split == 'train' or self.split == 'train_auto_quan' or self.split == 'val_auto_quan':
            return len(self.train_data)
        else:
            return len(self.test_data)

    def getAllData(self, split):
        data = None
        for file in files:
            data2 = np.loadtxt(self.root + '/' + split + '/Inertial Signals/' + file + '_' + split + '.txt', dtype=float)
            if data is None:
                data = data2
            else:
                data = np.concatenate((data, data2), axis=1)
        return data

    def getDataLabels(self, split):
        data = np.loadtxt(self.root + '/' + split + '/y_' + split + '.txt', dtype=int)
        return data

files = [
    'body_acc_x',
    'body_acc_y',
    'body_acc_z',
    'body_gyro_x',
    'body_gyro_y',
    'body_gyro_z',
    'total_acc_x',
    'total_acc_y',
    'total_acc_z'
]

if __name__ == '__main__':
    data_root = os.path.dirname(os.path.realpath(__file__)) + '/../data'
    data_loader = ActivityRecognitionDataset(root=os.path.join(data_root, 'UCI HAR Dataset'),
                       split='train',
                       transform=None,
                       target_transform=None)
    pass

