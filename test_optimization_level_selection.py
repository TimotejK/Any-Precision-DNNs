import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim
import torch.utils.data

import models
from datasets.data import get_dataset, get_transform
from models.losses import CrossEntropyLossSoft
from optimization_selection.optimization_selector import OptimizationSelector
from utils import setup_gpus

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


def test_optimization_selector(optimization_selector: OptimizationSelector):
    best_gpu = setup_gpus()
    torch.cuda.set_device(best_gpu)
    torch.backends.cudnn.benchmark = True

    val_transform = get_transform(args.dataset, 'val')
    val_data = get_dataset(args.dataset, 'val', val_transform)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    bit_width_list = list(map(int, args.bit_width_list.split(',')))
    bit_width_list.sort()
    model = models.__dict__[args.model](bit_width_list, val_data.num_classes).cuda()

    if args.pretrain and args.pretrain != 'None':
        if os.path.isdir(args.pretrain):
            args.pretrain = os.path.join(args.pretrain, 'model_best.pth.tar')
        if os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location='cuda:{}'.format(best_gpu))
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("loaded pretrain checkpoint '%s' (epoch %s)", args.pretrain, checkpoint['epoch'])
        else:
            raise ValueError('Pretrained model path error!')
    else:
        raise Exception("--pretrain argument has to be defined")

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    criterion = nn.CrossEntropyLoss().cuda()
    criterion_soft = CrossEntropyLossSoft().cuda()

    model.eval()
    optimization_selector.init(bit_width_list)

    ca = 0
    n = 0

    for i, (input, target) in enumerate(val_loader):
        val_data.__getitem__(i)
        with torch.no_grad():
            selection = optimization_selector.select_optimization_level(input[0], val_data.get_features(i))
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            model.apply(lambda m: setattr(m, 'wbit', selection))
            model.apply(lambda m: setattr(m, 'abit', selection))
            output = model(input)
            loss = criterion(output, target)
            prob, top_class = nnf.softmax(output, dim=1).topk(1, dim=1)

            confidence = prob[0][0]
            optimization_selector.results(confidence)
            if top_class[0][0] == target[0]:
                ca += 1
            n += 1
    print("ca:", ca/n)






if __name__ == '__main__':
    test_optimization_selector(OptimizationSelector())
