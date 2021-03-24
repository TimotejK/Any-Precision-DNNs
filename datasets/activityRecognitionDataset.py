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
    if len(row) > size ** 2:
        row = row[:size ** 2]
    number_of_rows = math.ceil(len(row) / size)
    padded_row = np.zeros(size * number_of_rows)
    padded_row[:len(row)] = row
    rows = padded_row.reshape(number_of_rows, size)
    result = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        j = int(number_of_rows * i / size)
        result[i, :] = rows[j]
    return result


def reshape_data(rows, labels):
    new_X = []
    new_labels = []
    previous_label = 0
    current_image = None
    for label, row in zip(labels, rows):
        if previous_label != label or current_image is None:
            current_image = [reshape_row(row)]
            previous_label = label
        else:
            current_image.append(reshape_row(row))
        if len(current_image) == 3:
            new_X.append(current_image)
            new_labels.append(label)
            current_image = None
    return np.array(new_X), np.array(new_labels)


def reshape_data_xyz_color(rows, labels):
    new_X = []
    new_labels = []
    previous_label = 0

    body_acc_x_combined = None
    body_acc_y_combined = None
    body_acc_z_combined = None
    body_gyro_x_combined = None
    body_gyro_y_combined = None
    body_gyro_z_combined = None
    total_acc_x_combined = None
    total_acc_y_combined = None
    total_acc_z_combined = None

    for label, row in zip(labels, rows):
        body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z, total_acc_x, total_acc_y, total_acc_z = np.split(
            row, 9)
        if previous_label != label or body_acc_x_combined is None:
            body_acc_x_combined, body_acc_y_combined, body_acc_z_combined, body_gyro_x_combined, body_gyro_y_combined, body_gyro_z_combined, total_acc_x_combined, total_acc_y_combined, total_acc_z_combined = \
                body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z, total_acc_x, total_acc_y, total_acc_z
            previous_label = label
        else:
            body_acc_x_combined = np.concatenate((body_acc_x_combined, body_acc_x))
            body_acc_y_combined = np.concatenate((body_acc_y_combined, body_acc_y))
            body_acc_z_combined = np.concatenate((body_acc_z_combined, body_acc_z))
            body_gyro_x_combined = np.concatenate((body_gyro_x_combined, body_gyro_x))
            body_gyro_y_combined = np.concatenate((body_gyro_y_combined, body_gyro_y))
            body_gyro_z_combined = np.concatenate((body_gyro_z_combined, body_gyro_z))
            total_acc_x_combined = np.concatenate((total_acc_x_combined, total_acc_x))
            total_acc_y_combined = np.concatenate((total_acc_y_combined, total_acc_y))
            total_acc_z_combined = np.concatenate((total_acc_z_combined, total_acc_z))
        if len(body_acc_x_combined) * 3 >= 32*32:
            new_X.append([reshape_row(np.concatenate((body_acc_x_combined, body_gyro_x_combined, total_acc_x_combined))),
                          reshape_row(np.concatenate((body_acc_y_combined, body_gyro_y_combined, total_acc_y_combined))),
                          reshape_row(np.concatenate((body_acc_z_combined, body_gyro_z_combined, total_acc_z_combined)))])
            new_labels.append(label)
            body_acc_x_combined = None
            body_acc_y_combined = None
            body_acc_z_combined = None
            body_gyro_x_combined = None
            body_gyro_y_combined = None
            body_gyro_z_combined = None
            total_acc_x_combined = None
            total_acc_y_combined = None
            total_acc_z_combined = None

    return np.array(new_X), np.array(new_labels)


class ActivityRecognitionDataset(data.Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        # now load the picked numpy arrays
        if self.split == 'train' or self.split == 'train_auto_quan' or self.split == 'val_auto_quan':
            self.train_data = self.getAllData("train")
            self.train_labels = self.getDataLabels("train")
            self.train_data, self.train_labels = reshape_data_xyz_color(self.train_data, self.train_labels)
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.train_labels = [int(x) - 1 for x in self.train_labels]
            pass
        else:
            self.test_data = self.getAllData("test")
            self.test_labels = self.getDataLabels("test")
            self.test_data, self.test_labels = reshape_data_xyz_color(self.test_data, self.test_labels)
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.test_labels = [int(x) - 1 for x in self.test_labels]

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
            data2 = np.loadtxt(self.root + '/' + split + '/Inertial Signals/' + file + '_' + split + '.txt',
                               dtype=float)
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
