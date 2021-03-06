import copy

from utils.reward_shaping.additive_functions import *
from utils.reward_shaping.aggregation_functions import *
from collections import deque

from scipy.misc import imresize
from skimage import color

DEFAULT_SEED_OFFSET = 2237
DEFAULT_SEED_MODULO = 1000000007


class Rewarder:
    def __init__(self, additive_functions_config, aggregation_functions_config, observation):
        self.additive_functions = []
        for cls in additive_functions_config:
            self.additive_functions.append(globals()[cls](additive_functions_config[cls]))
        self.aggregation_functions = []
        for f_config in aggregation_functions_config:
            self.aggregation_functions.append(globals()[f_config['class']](f_config['config']))
        self.previous_observation = None
        self.reset(observation)

    def reset(self, observation):
        self.previous_observation = copy.deepcopy(observation)
        for pf in self.additive_functions:
            pf.reset()
        for af in self.aggregation_functions:
            af.reset()

    def reward(self, observation, transformed_observation, current_reward, action):
        for aggregation_function in self.aggregation_functions:
            current_reward = aggregation_function(self.previous_observation, observation, current_reward)
        for additive_function in self.additive_functions:
            current_reward += additive_function(observation, transformed_observation, action)
        self.previous_observation = copy.deepcopy(observation)
        return current_reward


class PreliminaryTransformer:
        def __init__(self, shape=(120, 160, 3)):
            self.shape = shape

        def reset(self, observation):
            pass

        def transform(self, observation):
            resized = imresize(observation, self.shape)
            gray = color.rgb2gray(resized).reshape(self.shape[:2])
            return np.expand_dims(gray, axis=0)  # First dimension represents layers in torch


class Transformer:
    def __init__(self, repeat_obsservations=3, shape=(120, 160, 3)):
        self.repeat_observations = max(repeat_obsservations, 1)
        self.shape = shape
        self.previous_observations = deque(maxlen=self.repeat_observations)

    def reset(self, observation):
        self.previous_observations.clear()
        for _ in range(self.repeat_observations):
            self.previous_observations.append(observation)

    def transform(self, observation):
        observs = self.previous_observations
        observs.append(observation)
        while len(observs) > self.repeat_observations:
            observs.pop()
        return np.concatenate(observs, axis=0)