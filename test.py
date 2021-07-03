import argparse
import csv
import math
import os
import time
import socket
import logging
from datetime import datetime
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.nn.functional as nnf

import models
from models.losses import CrossEntropyLossSoft
from datasets.data import get_dataset, get_transform
from optimizer import get_optimizer_config, get_lr_scheduler
from local_utils import setup_logging, setup_gpus, save_checkpoint
from local_utils import AverageMeter, accuracy

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--results-dir', default='./results', help='results dir')
parser.add_argument('--dataset', default='imagenet', help='dataset name or folder')
parser.add_argument('--train_split', default='train', help='train split name')
parser.add_argument('--model', default='resnet18', help='model architecture')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
parser.add_argument('--optimizer', default='sgd', help='optimizer function used')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--lr_decay', default='100,150,180', help='lr decay steps')
parser.add_argument('--weight-decay', default=3e-4, type=float, help='weight decay')
parser.add_argument('--print-freq', '-p', default=20, type=int, help='print frequency')
parser.add_argument('--pretrain', default=None, help='path to pretrained full-precision checkpoint')
parser.add_argument('--resume', default=None, help='path to latest checkpoint')
parser.add_argument('--bit_width_list', default='4', help='bit width list')
args = parser.parse_args()


def main():
    hostname = socket.gethostname()
    setup_logging(os.path.join(args.results_dir, 'log_{}.txt'.format(hostname)))
    logging.info("running arguments: %s", args)

    best_gpu = setup_gpus()
    torch.cuda.set_device(best_gpu)
    torch.backends.cudnn.benchmark = True

    val_transform = get_transform(args.dataset, 'val')
    val_data = get_dataset(args.dataset, 'test+val', val_transform)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    bit_width_list = list(map(int, args.bit_width_list.split(',')))
    bit_width_list.sort()
    model = models.__dict__[args.model](bit_width_list, val_data.num_classes).cuda()

    lr_decay = list(map(int, args.lr_decay.split(',')))
    optimizer = get_optimizer_config(model, args.optimizer, args.lr, args.weight_decay)
    lr_scheduler = None
    best_prec1 = None
    if args.resume and args.resume != 'None':
        if os.path.isdir(args.resume):
            args.resume = os.path.join(args.resume, 'model_best.pth.tar')
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(best_gpu))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler = get_lr_scheduler(args.optimizer, optimizer, lr_decay, checkpoint['epoch'])
            logging.info("loaded resume checkpoint '%s' (epoch %s)", args.resume, checkpoint['epoch'])
        else:
            raise ValueError('Pretrained model path error!')
    elif args.pretrain and args.pretrain != 'None':
        if os.path.isdir(args.pretrain):
            args.pretrain = os.path.join(args.pretrain, 'model_best.pth.tar')
        if os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location='cuda:{}'.format(best_gpu))
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("loaded pretrain checkpoint '%s' (epoch %s)", args.pretrain, checkpoint['epoch'])
        else:
            raise ValueError('Pretrained model path error!')
    if lr_scheduler is None:
        lr_scheduler = get_lr_scheduler(args.optimizer, optimizer, lr_decay)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    criterion = nn.CrossEntropyLoss().cuda()
    criterion_soft = CrossEntropyLossSoft().cuda()
    sum_writer = SummaryWriter(args.results_dir + '/summary')

    for epoch in range(args.start_epoch, args.epochs):
        model.eval()
        bit_width_list = list(map(int, args.bit_width_list.split(',')))
        bit_width_list.sort()
        extract_classification_data(val_loader, model, criterion, bit_width_list, val_data)
        for width in bit_width_list:
            accs = run_inference(val_loader, model, criterion, width, val_data)
            print(width, accs, sum(accs)/6)


def run_inference(data_loader, model, criterion, bit_width, val_data):
    average_ac = AverageMeter()
    average_t5 = AverageMeter()
    acc = [0, 0, 0, 0, 0, 0]
    n = [0, 0, 0, 0, 0, 0]
    for i, (input, target) in enumerate(data_loader):
        val_data.__getitem__(i)
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            model.apply(lambda m: setattr(m, 'wbit', bit_width))
            model.apply(lambda m: setattr(m, 'abit', bit_width))
            model.apply(lambda m: setattr(m, 'width_mult', bit_width))
            output = model(input)
            loss = criterion(output, target)
            prob, top_class = nnf.softmax(output, dim=1).topk(1, dim=1)
            for p, t in zip(top_class, target):
                n[t] += 1
                if p[0] == t:
                    acc[t] += 1
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            average_ac.update(prec1.item())
            average_t5.update(prec5.item())

    # for i in range(len(n)):
    #     if (n[i] == 0):
    #         print(i, ":  inf")
    #     else:
    #         print(i, ": ", acc[i]/n[i])
    return np.array(acc) / np.array(n)


def extract_classification_data(data_loader, model, criterion, bit_width_list, val_data):
    with open('raw_data_accuracy_raw.csv', mode='w', newline='') as accuracy_file_raw, \
            open('raw_data_accuracy_features.csv', mode='w', newline='') as accuracy_file_features:
        field_names = ["raw " + str(i + 1) for i in range(32*32*3)]
        field_names.append("user")
        field_names.append("class")
        for bit_width in bit_width_list:
            field_names.append(str(bit_width) + " bit")
            field_names.append("confidence " + str(bit_width) + " bit")
        accuracy_file_writer_raw = csv.writer(accuracy_file_raw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        accuracy_file_writer_raw.writerow(field_names)

        field_names = list(val_data.get_feature_names())
        field_names.append("user")
        field_names.append("class")
        for bit_width in bit_width_list:
            field_names.append(str(bit_width) + " bit")
            field_names.append("confidence " + str(bit_width) + " bit")
        accuracy_file_writer_features = csv.writer(accuracy_file_features, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        accuracy_file_writer_features.writerow(field_names)

        for i, (input, target) in enumerate(data_loader):
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda(non_blocking=True)
                rows = [None] * data_loader.batch_size
                rows_features = [None] * data_loader.batch_size
                for bit_width in bit_width_list:
                    model.apply(lambda m: setattr(m, 'wbit', bit_width))
                    model.apply(lambda m: setattr(m, 'abit', bit_width))
                    model.apply(lambda m: setattr(m, 'width_mult', bit_width))
                    output = model(input)
                    loss = criterion(output, target)
                    prob, top_class = nnf.softmax(output, dim=1).topk(1, dim=1)
                    for j, (p, t) in enumerate(zip(top_class, target)):
                        if rows[j] is None:
                            rows[j] = input[j].reshape(32*32*3).tolist()
                            rows[j].append(val_data.get_user(i * data_loader.batch_size + j))
                            rows[j].append(int(t))
                        if rows_features[j] is None:
                            rows_features[j] = list(val_data.get_features(i * data_loader.batch_size + j))
                            rows_features[j].append(val_data.get_user(i * data_loader.batch_size + j))
                            rows_features[j].append(int(t))
                        if p[0] == t:
                            rows[j].append(1)
                            rows_features[j].append(1)
                        else:
                            rows[j].append(0)
                            rows_features[j].append(0)
                        rows[j].append(float(prob[j][0]))
                        rows_features[j].append(float(prob[j][0]))
                rows = filter(lambda x: x is not None, rows)
                rows_features = filter(lambda x: x is not None, rows_features)
                accuracy_file_writer_raw.writerows(rows)
                accuracy_file_writer_features.writerows(rows_features)

if __name__ == '__main__':
    main()
