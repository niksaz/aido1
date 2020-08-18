import copy

from features.straight.straight import line_approx
from utils.reward_shaping.additive_functions import *
from utils.reward_shaping.aggregation_functions import *
from collections import deque

import numpy as np
from PIL import Image

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

import cv2
class PreliminaryTransformer:
    def __init__(self, shape=(64, 48), use_segmentation=True, colorspace_conversion=None):
        """
        :param bool use_segmentation: whether to use or not segmentation (only works in simulator)
        :param colorspace_conversion: one of cv::ColorConversionCodes (like cv2.COLOR_RGB2HSV) or None
        """
        self.shape = shape
        self.use_segmentation = use_segmentation
        self.colorspace_conversion = colorspace_conversion

    def reset(self, observation):
        pass

    def transform(self, obs):
        if self.colorspace_conversion:
            obs = cv2.cvtColor(obs, self.colorspace_conversion)  # convert to desired colorspace

        if self.use_segmentation:
            obs = line_approx(np.array(obs, dtype=np.uint8))

        frame_lines_resized = np.array(Image.fromarray(obs).resize(self.shape))
        height = frame_lines_resized.shape[0]
        frame_lines_clipped = frame_lines_resized[height // 3:height]
        if frame_lines_clipped.ndim == 2:
            frame_lines_clipped = np.expand_dims(frame_lines_clipped, axis=-1)  # add layer dimension

        frame_lines_channeled = np.transpose(frame_lines_clipped, axes=(2, 0, 1))  # First dimension is for layers in torch
        return frame_lines_channeled


class Transformer:
    def __init__(self, repeat_observations=4):
        self.repeat_observations = max(repeat_observations, 1)
        self.previous_observations = deque(maxlen=self.repeat_observations)

    def reset(self, observation):
        self.previous_observations.clear()
        for _ in range(self.repeat_observations):
            self.previous_observations.append(observation)

    def transform(self, observation):
        observations = self.previous_observations
        observations.append(observation)
        while len(observations) > self.repeat_observations:
            observations.pop()
        return np.concatenate(observations, axis=0) / 255.0


class ActionTransformer:
    def __init__(self,
                 gain=1.0,
                 trim=0.0,
                 radius=0.0318,
                 k=27.0,
                 limit=1.0,
                 wheel_dist=0.102):
        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

        self.wheel_dist = wheel_dist

    def transform(self, action):
        # This is needed because at max speed the duckie can't turn anymore
        action = [action[0] * 0.8, action[1]]

        # Converts [velocity|heading] actions to [wheelvel_left|wheelvel_right]
        vel, angle = action

        # Assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # Adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * self.wheel_dist) / self.radius
        omega_l = (vel - 0.5 * angle * self.wheel_dist) / self.radius

        # Conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # Limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])
        return vels
