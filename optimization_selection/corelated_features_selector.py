import os

import torch
import torch.nn.functional as nnf
import numpy as np
from optimization_selection.optimization_selector import OptimizationSelector
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class CorelatedFeaturesSelector(OptimizationSelector):
    def __init__(self, n_groups=3):
        self.confidence = []
        self.train_acc = []
        self.n_groups = n_groups
        self.show_statistics = True
        self.target_acc = 0.9

    def init(self, optimization_levels):
        self.optimization_levels = optimization_levels[:3]

    def train(self, train_data_loader, train_data, model):
        groups = self.train_subdimention_models(train_data_loader, train_data)
        self.get_confidence_and_ca(train_data_loader, model)
        acc_per_group = self.accuracy_per_group(self.train_acc, groups)
        self.train_hierarchical_model(acc_per_group)

    def select_optimization_level(self, raw_data: list, features: list) -> float:
        group = self.transform_vector(list(features))

        if self.intercepts[int(group)] >= self.target_acc:
            selected_width = self.optimization_levels[0]
        else:
            selected_width = (self.target_acc - self.intercepts[group]) / self.slopes[group]

        return self.closest_available_width(selected_width)

    def transform_vector(self, feature_vector):
        return self.get_group(feature_vector[78])

    def train_subdimention_models(self, train_loader, train_data):
        data = [train_data.get_features(i)[78] for i in range(len(train_data))]
        return self.train_groups(data)

    def get_confidence_and_ca(self, train_data_loader, model):
        self.confidence = []
        self.train_acc = []
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
                    output = model(input)
                    prob, top_class = nnf.softmax(output, dim=1).topk(1, dim=1)
                    for ind, (p, c, t) in enumerate(zip(prob, top_class, target)):
                        confidence = p[0]
                        accuracy = 1 if c[0] == t else 0
                        if ind >= len(acc):
                            acc.append([])
                            self.confidence.append(float(confidence))
                        acc[ind].append(accuracy)

                self.train_acc += acc
        self.train_acc = np.array(self.train_acc)
        self.target_acc = np.mean(self.train_acc)
        self.confidence = np.array(self.confidence)

    def train_groups(self, values):
        sorted_values = values.copy()
        sorted_values.sort()
        self.borders = []
        for i in range(1, self.n_groups + 1):
            self.borders.append(sorted_values[int(i * (len(sorted_values) - 1) / self.n_groups)])
        groups = []
        for i in range(len(values)):
            groups.append(self.get_group(values[i]))
        return np.array(groups)

    def get_group(self, value):
        for i in range(self.n_groups):
            if value <= self.borders[i]:
                return i
        return self.n_groups - 1

    def accuracy_per_group(self, test_acc, groups):
        accuracy = []
        for g in range(self.n_groups):
            accuracy.append(np.sum(test_acc[groups == g], axis=0) / test_acc[groups == 0].shape[0])
        return np.array(accuracy)

    def train_hierarchical_model(self, accuracy_per_group):
        intercepts = []
        coefs = []
        for group in range(self.n_groups):
            model = LinearRegression()
            y = accuracy_per_group[group, :]
            x = np.array(self.optimization_levels).reshape((-1,1))
            model.fit(x, y)
            intercepts.append(model.intercept_)
            coefs.append(model.coef_)
        self.intercepts = np.array(intercepts)
        self.slopes = np.array(coefs)
        if self.show_statistics:
            self.show_model(accuracy_per_group)

    def show_model(self, accuracy_per_group):
        training_set = []
        for group in range(accuracy_per_group.shape[0]):
            for bit_width in range(accuracy_per_group.shape[1]):
                training_set.append(
                    [self.optimization_levels[bit_width], group, accuracy_per_group[group, bit_width]])
        training_set = np.array(training_set)
        # draw model
        colors = ['b', 'r', 'g', 'y', 'm', 'c']
        for point in training_set:
            plt.plot(float(point[0]), float(point[2]), colors[int(point[1])] + 'o')

        for i in range(len(self.intercepts)):
            x = np.linspace(0, np.max(training_set[:, 0]), 10)
            plt.plot(x, self.intercepts[i] + self.slopes[i] * x, colors[i] + '')
        plt.title("Hierarchical model")
        plt.xlabel("Bit width")
        plt.ylabel("Classification accuracy")
        plt.show()

    def closest_available_width(self, width):
        closest_width = self.optimization_levels[0]
        distance = abs(self.optimization_levels[0] - width)
        for w in self.optimization_levels:
            d = abs(width - w)
            if d < distance:
                distance = d
                closest_width = w
        return closest_width
