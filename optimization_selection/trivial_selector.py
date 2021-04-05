from typing import List

from optimization_selection.optimization_selector import OptimizationSelector


class ConstantSelector(OptimizationSelector):
    def __init__(self, constant_width):
        self.constant_width = constant_width

    def select_optimization_level(self, raw_data: list, features: list) -> float:
        return self.constant_width

