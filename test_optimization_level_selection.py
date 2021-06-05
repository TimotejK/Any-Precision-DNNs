import argparse
import copy
import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim
import torch.utils.data

import models
from datasets.activityRecognitionDataset import ActivityRecognitionDataset
from datasets.data import get_dataset, get_transform
from models.losses import CrossEntropyLossSoft
from optimization_selection.confidence_hierarchical_selector import ConfidenceHierarchicalSelector
from optimization_selection.confidence_simple_selector import ConfidenceSimpleSelector
from optimization_selection.corelated_features_selector import CorelatedFeaturesSelector
from optimization_selection.knn_selector import KnnSelector
from optimization_selection.lda_hierarchical_selector import LDAHierarchicalSelector
from optimization_selection.lda_hierarchical_selector2 import LDAAccuracySelector
from optimization_selection.optimization_selector import OptimizationSelector
from optimization_selection.random_selector import RandomSelector
from optimization_selection.trivial_selector import ConstantSelector
if os.environ['CPU'] != 'True':
    from utils.utils import setup_gpus

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

data_root = os.path.dirname(os.path.realpath(__file__)) + '/data'

selection_cost = {
    1: 1,
    2: 2,
    4: 3,
    8: 4,
    32: 5
}


def test_optimization_selector_CV(optimization_selector: OptimizationSelector):
    test_users = [2, 4, 9, 10, 12, 13, 18, 20, 24]

    average_ca = 0
    average_conf = 0
    average_selection = 0
    average_algorithm_percentage = 0
    average_algorithm_time = 0
    n = 0

    for i in range(len(test_users)):
        val_transform = get_transform(args.dataset, 'val')
        val_data = ActivityRecognitionDataset(root=os.path.join(data_root, 'UCI HAR Dataset'),
                                              split='test',
                                              transform=val_transform,
                                              target_transform=None,
                                              selected_users=test_users[:i] + test_users[i + 1:])
        val_data.num_classes = 6

        test_data = ActivityRecognitionDataset(root=os.path.join(data_root, 'UCI HAR Dataset'),
                                               split='test',
                                               transform=val_transform,
                                               target_transform=None,
                                               selected_users=[test_users[i]])
        test_data.num_classes = 6

        (ca, conf, selection, selection_theoretical, algorithm_time, algorithm_percentage) = \
            test_optimization_selector(copy.deepcopy(optimization_selector), val_data, test_data)
        average_ca += ca
        average_conf += conf
        average_selection += selection
        average_algorithm_percentage += algorithm_percentage
        average_algorithm_time += algorithm_time
        n += 1

    print("ca:", average_ca / n)
    print("Confidence:", average_conf / n)
    print("average_selection:", average_selection / n)
    print("Algorithm time s:", average_algorithm_time / n)
    print("Selection time [%]:", average_algorithm_percentage / n)


def test_optimization_selector(optimization_selector: OptimizationSelector, val_data=None, test_data=None):
    if os.environ['CPU'] != 'True':
        best_gpu = setup_gpus()
        torch.cuda.set_device(best_gpu)
    torch.backends.cudnn.benchmark = True

    val_transform = get_transform(args.dataset, 'val')

    if val_data is None:
        val_data = get_dataset(args.dataset, 'val', val_transform)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=64,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if test_data is None:
        test_data = get_dataset(args.dataset, 'test', val_transform)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)

    bit_width_list = list(map(int, args.bit_width_list.split(',')))
    bit_width_list.sort()
    if os.environ['CPU'] != 'True':
        model = models.__dict__[args.model](bit_width_list, val_data.num_classes).cuda()
    else:
        model = models.__dict__[args.model](bit_width_list, val_data.num_classes)

    if args.pretrain and args.pretrain != 'None':
        if os.path.isdir(args.pretrain):
            args.pretrain = os.path.join(args.pretrain, 'model_best.pth.tar')
        if os.path.isfile(args.pretrain):
            if os.environ['CPU'] != 'True':
                checkpoint = torch.load(args.pretrain, map_location='cuda:{}'.format(best_gpu))
            else:
                checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("loaded pretrain checkpoint '%s' (epoch %s)", args.pretrain, checkpoint['epoch'])
        else:
            raise ValueError('Pretrained model path error!')
    else:
        raise Exception("--pretrain argument has to be defined")

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    if os.environ['CPU'] != 'True':
        criterion = nn.CrossEntropyLoss().cuda()
        criterion_soft = CrossEntropyLossSoft().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
        criterion_soft = CrossEntropyLossSoft()

    model.eval()

    model.apply(lambda m: setattr(m, 'true_quantization', True))

    optimization_selector.init(bit_width_list)
    optimization_selector.train(val_loader, val_data, model)

    ca = 0
    conf = 0
    n = 0
    selection_sum = 0
    selection_sum_theoretical = 0

    start_time = time.time()
    size_selection_time = 0.0
    for i, (input, target) in enumerate(test_loader):
        with torch.no_grad():
            repeat = True
            while repeat:
                selection_start_time = time.time()
                selection = optimization_selector.select_optimization_level(input[0], test_data.get_features(i))
                size_selection_time += time.time() - selection_start_time
                selection_sum += selection
                if os.environ['CPU'] != 'True':
                    input = input.cuda()
                    target = target.cuda(non_blocking=True)

                model.apply(lambda m: setattr(m, 'wbit', selection))
                model.apply(lambda m: setattr(m, 'abit', selection))
                output = model(input)
                loss = criterion(output, target)
                prob, top_class = nnf.softmax(output, dim=1).topk(1, dim=1)

                prediction = int(top_class[0][0])
                confidence = float(prob[0][0])
                selection_start_time = time.time()
                repeat = optimization_selector.results(prediction, confidence)
                size_selection_time += time.time() - selection_start_time
                if not repeat:
                    selection_sum_theoretical += selection_cost[selection]
            conf += float(confidence)
            if top_class[0][0] == target[0]:
                ca += 1
            n += 1
    algorithm_time = time.time() - start_time
    # print("ca:", ca / n)
    # print("Confidence:", conf / n)
    # print("average_selection:", selection_sum / n)
    # print("average_selection_theoretical:", selection_sum_theoretical / n)
    # print("Time:", algorithm_time)
    # print("Selection time [%]:", 100 * size_selection_time / algorithm_time)

    return (ca / n, conf / n, selection_sum / n, selection_sum_theoretical / n, algorithm_time,
            100 * size_selection_time / algorithm_time)


if __name__ == '__main__':
    # print("Constant 1:")
    # test_optimization_selector_CV(ConstantSelector(1))
    # print("Constant 2:")
    # test_optimization_selector_CV(ConstantSelector(2))
    # print("Constant 4:")
    # test_optimization_selector_CV(ConstantSelector(4))
    # print("Constant 8:")
    # test_optimization_selector_CV(ConstantSelector(8))
    print("Constant 32:")
    test_optimization_selector_CV(ConstantSelector(32))
    # print("Random:")
    # test_optimization_selector_CV(RandomSelector())
    # print("Random 1-4:")
    # test_optimization_selector_CV(RandomSelector([1, 2, 4]))
    # print("Knn 20:")
    # test_optimization_selector_CV(KnnSelector(20))
    # print("Knn 40:")
    # test_optimization_selector_CV(KnnSelector(40))
    # print("Knn 100:")
    # test_optimization_selector_CV(KnnSelector(100))
    # print("Confidence hierarchical:")
    # test_optimization_selector_CV(ConfidenceHierarchicalSelector())
    # print("LDA Accuracy hierarchical:")
    # test_optimization_selector_CV(LDAAccuracySelector(use_features=True))
    # print("Simple conficence:")
    # test_optimization_selector_CV(ConfidenceSimpleSelector())
    # print("LDA hierarchical features:")
    # test_optimization_selector_CV(LDAHierarchicalSelector(use_features=True, n_groups=5))
    # print("LDA hierarchical raw signal:")
    # test_optimization_selector_CV(LDAHierarchicalSelector(use_features=False))
    # print("Best feature subspace:")
    # test_optimization_selector_CV(CorelatedFeaturesSelector(n_groups=10))
    pass
