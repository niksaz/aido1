import abc

import numpy as np


class RewardAggregationFunction:
    @abc.abstractclassmethod
    def __init__(self, config):
        pass

    @abc.abstractclassmethod
    def __call__(self, previous_observation: dict, current_observation: dict, current_reward):
        pass

    @abc.abstractclassmethod
    def reset(self):
        pass


class BaselineAggregationFunction(RewardAggregationFunction):
    def __init__(self, config):
        super().__init__(config)

    def reset(self):
        pass

    def __call__(self, previous_observation: dict, current_observation: dict, current_reward):
        if current_reward == -1000:
            current_reward = -10
        elif current_reward > 0:
            current_reward += 10
        else:
            current_reward += 4
        return current_reward