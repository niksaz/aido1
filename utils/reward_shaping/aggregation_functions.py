import abc

import numpy as np


class RewardAggregationFunction:
    @abc.abstractclassmethod
    def __call__(self, previous_observation: dict, current_observation: dict, current_reward):
        pass

    @abc.abstractclassmethod
    def reset(self):
        pass


class TransformAndBound(RewardAggregationFunction):
    def __init__(self, config):
        self.scale = config.get('scale', 1.)
        self.move = config.get('move', 0.)
        self.bound = config.get('bound', 1e8)
        self.lower_bound = config.get('lower_bound', -self.bound)

    def reset(self):
        pass

    def __call__(self, previous_observation: dict, current_observation: dict, current_reward):
        return min(max(current_reward * self.scale + self.move, self.lower_bound), self.bound)


class NormalizeAggFunction(RewardAggregationFunction):
    def reset(self):
        pass

    def __init__(self, config):
        self.scale_range = config['scale_range']
        self.move_range = config['move_range']
        self.remove_activations_penalty = config.get('remove_activation_penalty', False)
        self.bound = config.get("bound", False)

    def __call__(self, previous_observation: dict, current_observation: dict, current_reward):
        factor = max(np.linalg.norm(current_observation['target_vel'])**2, 0.1)
        if self.remove_activations_penalty:
            current_reward = 10. - sum((np.array(current_observation['target_vel'])[[0, 2]] -
                                        np.array(current_observation['body_vel']['pelvis'])[[0, 2]])**2)
            penalty = 0.
        else:
            penalty = current_reward - (10. - sum((np.array(current_observation['target_vel'])[[0, 2]] -
                                        np.array(current_observation['body_vel']['pelvis'])[[0, 2]])**2))
            current_reward -= penalty
        if self.move_range:
            current_reward -= (10. - factor)
        if self.bound:
            if self.move_range:
                current_reward = max(-factor, current_reward)
            else:
                current_reward = max(0, current_reward)
        if self.scale_range:
            if self.move_range:
                current_reward /= factor
            else:
                current_reward /= 10.
        current_reward += penalty
        return current_reward
