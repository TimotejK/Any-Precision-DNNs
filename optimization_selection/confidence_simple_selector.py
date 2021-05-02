import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as nnf

from optimization_selection.optimization_selector import OptimizationSelector


class ConfidenceSimpleSelector(OptimizationSelector):
    def __init__(self):
        self.alpha = 0.1

    def init(self, optimization_levels: List[int]):
        self.optimization_levels = optimization_levels
        self.selected_optimization = optimization_levels[0]
        self.previous_activity = -1

    def train(self, train_data_loader, train_data, model):
        self.get_confidence_and_ca(train_data_loader, model)

    def select_optimization_level(self, raw_data: list, features: list) -> float:
        return self.selected_optimization

    def results(self, predicition: int, confidence: float) -> bool:
        activity_index = np.where(self.unique_activities == predicition)[0]
        if confidence < self.confidence_thresholds[activity_index]:
            new_selected_index = np.where(self.optimization_levels == self.selected_optimization)[0] + 1
            if new_selected_index >= len(self.optimization_levels):
                return False
            self.selected_optimization = self.optimization_levels[new_selected_index]
            return True

        if predicition != self.previous_activity:
            self.selected_optimization = self.optimization_levels[0]
        elif random.randint(0, 100) < 100 * self.alpha:
            self.selected_optimization = self.optimization_levels[0]

        self.previous_activity = predicition
        return False

    def get_confidence_and_ca(self, train_data_loader, model):
        self.confidence = []
        self.train_acc = []
        self.activity = []
        for i, (input, target) in enumerate(train_data_loader):
            with torch.no_grad():
                acc = []
                conf = []
                for bit_width in self.optimization_levels:
                    input = input.cuda()
                    target = target.cuda(non_blocking=True)
                    model.apply(lambda m: setattr(m, 'wbit', bit_width))
                    model.apply(lambda m: setattr(m, 'abit', bit_width))
                    output = model(input)
                    prob, top_class = nnf.softmax(output, dim=1).topk(1, dim=1)
                    for ind, (p, c, t) in enumerate(zip(prob, top_class, target)):
                        confidence = p[0]
                        accuracy = 1 if c[0] == t else 0
                        if ind >= len(acc):
                            acc.append([])
                            conf.append([])
                            self.activity.append(int(t))
                        acc[ind].append(accuracy)
                        conf[ind].append(float(confidence))

                self.train_acc += acc
                self.confidence += conf
        self.train_acc = np.array(self.train_acc)
        self.target_acc = np.mean(self.train_acc)
        self.confidence = np.array(self.confidence)
        self.activity = np.array(self.activity)

        self.confidence_thresholds = []
        self.unique_activities = np.unique(self.activity)
        self.unique_activities.sort()
        for activity in self.unique_activities:
            accuracys = np.mean(self.train_acc[self.activity == activity], 0)
            index = accuracys.argmax()
            mean_confidence = np.mean(self.confidence[self.activity == activity], 0)[index]
            threshold = mean_confidence - 2 * np.std(self.confidence[self.activity == activity], 0)[index]
            self.confidence_thresholds.append(threshold)
        self.confidence_thresholds = np.array(self.confidence_thresholds)
