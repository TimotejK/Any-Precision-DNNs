import os

import torch
import torch.nn.functional as nnf
from scipy import spatial
import numpy as np

from optimization_selection.optimization_selector import OptimizationSelector


class KnnSelector(OptimizationSelector):
    def __init__(self, k=20):
        self.train_vectors = []
        self.train_acc = []
        self.k = k

    def train(self, train_data_loader, train_data, model):
        for i, (input, target) in enumerate(train_data_loader):
            with torch.no_grad():
                acc = []
                for bit_width in self.optimization_levels:
                    if os.environ['CPU'] != 'True':
                        input = input.cuda()
                        target = target.cuda(non_blocking=True)
                    else:
                        input = input
                        target = target
                    model.apply(lambda m: setattr(m, 'wbit', bit_width))
                    model.apply(lambda m: setattr(m, 'abit', bit_width))
                    model.apply(lambda m: setattr(m, 'width_mult', bit_width))
                    output = model(input)
                    prob, top_class = nnf.softmax(output, dim=1).topk(1, dim=1)
                    for ind, (p, c, t) in enumerate(zip(prob, top_class, target)):
                        confidence = p[0]
                        accuracy = 1 if c[0] == t else 0
                        if ind >= len(acc):
                            acc.append([])
                            features = train_data.get_features(i * train_data_loader.batch_size + ind)
                            self.train_vectors.append(features)

                        acc[ind].append(accuracy)

                self.train_acc += acc
        self.train_acc = np.array(self.train_acc)
        self.train_vectors_kd_tree = spatial.KDTree(self.train_vectors)

    def select_optimization_level(self, raw_data: list, features: list) -> float:
        k = self.k
        _, nearest_neighbours = self.train_vectors_kd_tree.query(features, k=k)
        acc = np.array([0 for _ in range(len(self.optimization_levels))])
        for nn in nearest_neighbours:
            acc += self.train_acc[nn]
        selected_index = acc.argmax()
        return self.optimization_levels[selected_index]

    pass
