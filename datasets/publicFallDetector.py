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
    std = 6.0
    row = np.array(list(map(lambda x: clamp(128 + (x / (2 * std)) * 128, 0, 255), row)))

    size = 32
    number_of_rows = math.ceil(len(row) / size)
    padded_row = np.zeros(size * number_of_rows)
    padded_row[:len(row)] = row
    rows = padded_row.reshape(number_of_rows, size)
    result = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        j = int(number_of_rows * i / size)
        result[i, :] = rows[j]
    return result


class PublicFallDetector(data.Dataset):

    def __init__(self, root, split, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        # now load the picked numpy arrays
        if self.split == 'train' or self.split == 'train_auto_quan' or self.split == 'val_auto_quan':
            self.train_data = [None, None, None]
            self.train_labels = None
            # take first 8 participants and leave last 2 for testing
            for axis in range(3):
                fall_data = self.getAllDataAsListNew("fall", "pocket", axis)[:8]
                adl_data = self.getAllDataAsListNew("adl", "pocket", axis)[:8]
                for person_data in fall_data:
                    self.train_data[axis] = person_data if self.train_data[axis] is None else np.concatenate(
                        (self.train_data[axis], person_data))
                    if axis == 0:
                        self.train_labels = np.ones(len(person_data)) if self.train_labels is None else np.concatenate(
                            (self.train_labels, np.ones(len(person_data))))
                for person_data in adl_data:
                    self.train_data[axis] = person_data if self.train_data[axis] is None else np.concatenate(
                        (self.train_data[axis], person_data))
                    if axis == 0:
                        self.train_labels = np.zeros(len(person_data)) if self.train_labels is None else np.concatenate(
                            (self.train_labels, np.zeros(len(person_data))))

            self.train_data = np.array([np.array(list(map(reshape_row, self.train_data[0]))),
                                        np.array(list(map(reshape_row, self.train_data[1]))),
                                        np.array(list(map(reshape_row, self.train_data[2])))])
            self.train_data = self.train_data.transpose((1, 2, 3, 0))  # convert to HWC
            self.train_labels = [int(x) for x in self.train_labels]
            pass
        else:
            self.test_data = [None, None, None]
            self.test_labels = None
            for axis in range(3):
                # take first 8 participants and leave last 2 for testing
                fall_data = self.getAllDataAsListNew("fall", "pocket", axis)[8:]
                adl_data = self.getAllDataAsListNew("adl", "pocket", axis)[8:]
                for person_data in fall_data:
                    self.test_data[axis] = person_data if self.test_data[axis] is None else np.concatenate(
                        (self.test_data[axis], person_data))
                    if axis == 0:
                        self.test_labels = np.ones(len(person_data)) if self.test_labels is None else np.concatenate(
                            (self.test_labels, np.ones(len(person_data))))
                for person_data in adl_data:
                    self.test_data[axis] = person_data if self.test_data[axis] is None else np.concatenate(
                        (self.test_data[axis], person_data))
                    if axis == 0:
                        self.test_labels = np.zeros(len(person_data)) if self.test_labels is None else np.concatenate(
                            (self.test_labels, np.zeros(len(person_data))))

            self.test_data = np.array([np.array(list(map(reshape_row, self.test_data[0]))),
                                       np.array(list(map(reshape_row, self.test_data[1]))),
                                       np.array(list(map(reshape_row, self.test_data[2])))])
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

    def getAllDataAsListNew(self, kind, position, axis):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        """
        Obtains data of all people together as a list (each member for a given person)
        Each entry is an array. We use the data in vectorial form to get only the total acceleration
        kind='fall' or 'adl'
        position='pocket' or 'hbag'
        Some combinations are not implemented yet
        Returns the list of data. Each element of the list is an array, in which each row is a temporal sequence
        of acceleration values
        """
        if (kind == 'fall' and position == 'pocket'):
            falldum = sc.loadtxt(self.root + '/person0/fallProcessedVector/0fallPV.dat')
            fall0 = falldum[axis::3]
            ###
            falldum = sc.loadtxt(self.root + '/person1/fallProcessedVector/1fallPV.dat')
            fall1 = falldum[axis::3]
            ###
            falldum = sc.loadtxt(self.root + '/person2/fallProcessedVector/2fallPV.dat')
            fall2 = falldum[axis::3]
            ###
            falldum = sc.loadtxt(self.root + '/person3/fallProcessedVector/3fallPV.dat')
            fall3 = falldum[axis::3]
            ###
            falldum = sc.loadtxt(self.root + '/person4/fallProcessedVector/4fallPV.dat')
            fall4 = falldum[axis::3]
            ###
            falldum = sc.loadtxt(self.root + '/person5/fallProcessedVector/5fallPV.dat')
            fall5 = falldum[axis::3]
            ###
            falldum = sc.loadtxt(self.root + '/person6/fallProcessedVector/6fallPV.dat')
            fall6 = falldum[axis::3]
            ###
            falldum = sc.loadtxt(self.root + '/person7/fallProcessedVector/7fallPV.dat')
            fall7 = falldum[axis::3]
            ###
            falldum = sc.loadtxt(self.root + '/person8/fallProcessedVector/8fallPV.dat')
            fall8 = falldum[axis::3]
            ###
            falldum = sc.loadtxt(self.root + '/person9/fallProcessedVector/9fallPV.dat')
            fall9 = falldum[axis::3]
            ###
            return (fall0, fall1, fall2, fall3, fall4, fall5, fall6, fall7, fall8, fall9)
            ####################
        elif (kind == 'adl' and position == 'pocket'):
            adldum = sc.loadtxt(self.root + '/person0/adlProcessedVector/0adlPV.dat')
            adl0 = adldum[axis::3]
            ###
            adldum = sc.loadtxt(self.root + '/person1/adlProcessedVector/1adlPV.dat')
            adl1 = adldum[axis::3]
            ###
            adldum = sc.loadtxt(self.root + '/person2/adlProcessedVector/2adlPV.dat')
            adl2 = adldum[axis::3]
            ###
            adldum = sc.loadtxt(self.root + '/person3/adlProcessedVector/3adlPV.dat')
            adl3 = adldum[axis::3]
            ###
            adldum = sc.loadtxt(self.root + '/person4/adlProcessedVector/4adlPV.dat')
            adl4 = adldum[axis::3]
            ####
            adldum = sc.loadtxt(self.root + '/person5/adlProcessedVector/5adlPV.dat')
            adl5 = adldum[axis::3]
            ###
            adldum = sc.loadtxt(self.root + '/person6/adlProcessedVector/6adlPV.dat')
            adl6 = adldum[axis::3]
            ###
            adldum = sc.loadtxt(self.root + '/person7/adlProcessedVector/7adlPV.dat')
            adl7 = adldum[axis::3]
            ###
            adldum = sc.loadtxt(self.root + '/person8/adlProcessedVector/8adlPV.dat')
            adl8 = adldum[axis::3]
            ###
            adldum = sc.loadtxt(self.root + '/person9/adlProcessedVector/9adlPV.dat')
            adl9 = adldum[axis::3]
            ###
            return (adl0, adl1, adl2, adl3, adl4, adl5, adl6, adl7, adl8, adl9)
        else:
            return ()


if __name__ == '__main__':
    data_root = os.path.dirname(os.path.realpath(__file__)) + '/../data'
    PublicFallDetector(root=os.path.join(data_root, 'data201307'),
                       split='train',
                       transform=None,
                       target_transform=None)
