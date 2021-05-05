import random
from typing import List

from optimization_selection.optimization_selector import OptimizationSelector


class RandomSelector(OptimizationSelector):
    def __init__(self, valid_selections=None):
        self.valid_selections = valid_selections

    def init(self, optimization_levels: List[int]):
        if self.valid_selections is None:
            self.valid_selections = optimization_levels

    def select_optimization_level(self, raw_data: list, features: list) -> float:
        return random.choice(self.valid_selections)

