from typing import List
import pickle as pk

class OptimizationSelector:
    savable_parameters = []

    def init(self, optimization_levels: List[int]):
        self.optimization_levels = optimization_levels
        pass

    def train(self, train_data_loader, train_data, model):
        pass

    def select_optimization_level(self, raw_data: list, features: list) -> float:
        return max(self.optimization_levels)

    def results(self, prediction:int, confidence: float) -> bool:
        # return if classification needs to be rerun
        return False

    def save(self, file_path):
        dict = {}
        for param in self.savable_parameters:
            dict[param] = self.__getattribute__(param)
        with open(file_path, "wb") as file:
            pk.dump(dict, file)

    def load(self, file_path):
        with open(file_path, "rb") as file:
            dict = pk.load(file)
        for param in self.savable_parameters:
            self.__setattr__(param, dict[param])

