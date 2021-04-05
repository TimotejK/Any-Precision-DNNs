from typing import List


class OptimizationSelector:
    def init(self, optimization_levels: List[int]):
        self.optimization_levels = optimization_levels
        pass

    def train(self, train_data_loader, train_data, model):
        pass

    def select_optimization_level(self, raw_data: list, features: list) -> float:
        return max(self.optimization_levels)

    def results(self, confidence: float) -> bool:
        # return if classification needs to be rerun
        return False
