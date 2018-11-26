import abc
import math

import numpy as np


class RewardAdditiveFunction:
    @abc.abstractmethod
    def __init__(self, config):
        pass

    @abc.abstractmethod
    def __call__(self, current_observation: dict, current_observation_transformed, action):
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class StayAliveAddF(RewardAdditiveFunction):
    def reset(self):
        pass

    def __call__(self, current_observation: dict, current_observation_transformed, action):
        return self.additive

    def __init__(self, config):
        super().__init__(config)
        self.additive = config["reward"]


class DecayStayAliveAddF(RewardAdditiveFunction):
    def reset(self):
        self.step += 1

    def __call__(self, current_observation: dict, current_observation_transformed, action):
        return self.start + (self.end - self.start) * self.step / self.steps

    def __init__(self, config):
        super().__init__(config)
        self.start = config["start"]
        self.end = config["end"]
        self.steps = config["steps"]
        self.step = 0


class SpeedAddF(RewardAdditiveFunction):
    def __init__(self, config):
        super().__init__(config)
        self.add_rewards = config['add_rewards']

    def __call__(self, current_observation: dict, current_observation_transformed, action):
        # pelvis_height = current_observation['body_pos']['pelvis'][1]
        # pelvis_height_target = 0.88
        # print('pelvis height: {}'.format(pelvis_height))
        # done = observation['body_pos']['pelvis'][1] < 0.3

        # torso_vel_rot = current_observation['body_vel_rot']['torso'][1]
        # print('torso_rot: {}'.format(torso_vel_rot))
        # print('pelvis_rot: {}'.format(observation['body_pos_rot']['pelvis']))

        add_reward = 0.

        for body_part in sorted(self.add_rewards):
            body_part_speed = current_observation['body_vel'][body_part][0]
            add_reward += 9 - np.abs(3 - body_part_speed) ** 2

        # pelvis_height_reward = np.abs(pelvis_height - pelvis_height_target) * 10
        # torso_vel_rot_reward = np.abs(torso_vel_rot)

        # reward_before = reward_with_mod
        #
        # if self.env_step < random.randint(24, 32):
        #     reward_with_mod = reward + sum(add_rewards.values()) - pelvis_height_reward - torso_vel_rot_reward

        # print(self.env_step, reward_before, reward_with_mod, pelvis_height_reward, torso_vel_rot_reward)
        return add_reward


class DeadlineAddF(RewardAdditiveFunction):
    def __init__(self, config):
        super().__init__(config)
        self.border = config['border']
        self.deadline = config.get('deadline', 0.6)
        self.max_reward = config['max_reward']
        self.kernel = config.get('kernel', 'hyperbolic')

    def __call__(self, current_observation: dict, current_observation_transformed, action):
        if current_observation['body_pos']['pelvis'][1] > self.border:
            return 0.
        x = (self.border - current_observation['body_pos']['pelvis'][1]) / (self.border - self.deadline)
        if self.kernel == 'arctan':
            d = 2 * np.arctan(20 * x**2) / math.pi
        elif self.kernel == 'logistic':
            d = 2 / (1 + np.exp(-20 * x**2)) - 1
        elif self.kernel == 'linear':
            d = x
        elif self.kernel == 'hyperbolic':
            d = 1 / (30 * abs(1 - x)**3 + 1)
        else:
            raise NotImplemented()
        return self.max_reward * d


class TiltPF(RewardAdditiveFunction):
    def __init__(self, config):
        super().__init__(config)
        self.min = config["min_reward"]
        self.max = config["max_reward"]
        self.body_len = None
        self.gamma = config["gamma"]
        self.potential = 0.

    def reset(self):
        self.potential = 0.

    def __call__(self, current_observation: dict, current_observation_transformed, action):
        if self.body_len is None:
            self.body_len = np.linalg.norm(np.array(current_observation["body_pos"]["pelvis"]) -
                                           np.array(current_observation["body_pos"]["head"]))
        potential = np.linalg.norm(np.array(current_observation["body_pos"]["pelvis"])[[0, 2]] -
                                   np.array(current_observation["body_pos"]["head"])[[0, 2]]) / self.body_len
        potential = self.max - (self.min - self.max) * potential
        reward = self.gamma * potential - self.potential
        self.potential = potential
        return reward


class SpeedDiffPF(RewardAdditiveFunction):
    def __init__(self, config):
        super().__init__(config)
        self.min_reward = config["min"]
        self.max_reward = config["max"]
        self.gamma = config["gamma"]
        self.kernel = config.get("kernel", "arctan")
        self.potential = 0

    def reset(self):
        self.potential = 0

    def __call__(self, current_observation: dict, current_observation_transformed, action):
        potential = self.get_reward_by_distance(np.linalg.norm(
            np.array(current_observation["target_vel"])[[0, 2]] -
            np.array(current_observation["body_vel"]["pelvis"])[[0, 2]]
        ) / np.linalg.norm(current_observation["target_vel"]))
        reward = self.gamma * potential - self.potential
        self.potential = potential
        return reward

    def get_reward_by_distance(self, distance):
        if self.kernel == 'arctan':
            d = 2 * np.arctan(6 * distance) / math.pi
        elif self.kernel == 'logistic':
            d = 2 / (1 + np.exp(-6 * distance)) - 1
        else:
            raise NotImplemented()
        return self.max_reward - (self.max_reward - self.min_reward) * d


class BasePosePF(RewardAdditiveFunction):
    def __init__(self, config):
        """
        Parameters
        ----------
            min_reward : Minimal reward that can be reached for similarity
            max_reward : Maximal reward that can be reached for similarity
            distance_scale : Scale distance
            gamma : coefficient that corresponds to model value-function gamma
            dataset : Path to data set file
            features : Feature names in order as in data set
            kernel : Applied to distance while computing additional reward ['arctan', 'logistic']. Default 'arctan'
            zero_point_feature : feature to which observation features will be normalized. Default 'head'
        """
        super().__init__(config)
        self.min_reward = config['min_reward']
        self.max_reward = config['max_reward']

        self.d_scale = config['distance_scale']
        self.features = config['features']
        self.kernel = config.get('kernel', 'arctan')
        self.normalization_feature = config.get('zero_point_feature', 'head')
        self.gamma = config.get('gamma', 1.)
        self.step_count = 0

        self.previous_reward = 0.

    def __call__(self, current_observation, current_observation_transformed, action):
        self.step_count += 1
        current_reward = self.get_reward(self.point_by_observation(current_observation))
        potential_reward = self.gamma * current_reward - self.previous_reward
        self.previous_reward = current_reward
        return potential_reward

    def normalize_observation(self, observation):
        body_pos = observation['body_pos']
        body_projection_length = 0.
        body_projection_length += np.sum(
            (np.array(body_pos['head'][:2]) - np.array(body_pos['femur_l'][:2])) ** 2) ** 0.5
        body_projection_length += np.sum(
            (np.array(body_pos['femur_l'][:2]) - np.array(body_pos['tibia_l'][:2])) ** 2) ** 0.5
        body_projection_length += np.sum(
            (np.array(body_pos['tibia_l'][:2]) - np.array(body_pos['talus_l'][:2])) ** 2) ** 0.5
        normalized_observation = {}
        for key in body_pos.keys():
            normalized_observation[key] = np.array(body_pos[key]) / body_projection_length
        norm_feature = normalized_observation[self.normalization_feature]
        for key in normalized_observation.keys():
            normalized_observation[key] = normalized_observation[key] - norm_feature

        return normalized_observation

    def point_by_observation(self, observation):
        norm_obs = self.normalize_observation(observation)
        x = []
        for feature in self.features:
            x.append(norm_obs[feature][0])
            x.append(norm_obs[feature][1])
        return x

    def get_reward_by_distances(self, distances):
        distances = np.array(distances)
        if self.kernel == 'arctan':
            d = 2 * np.arctan(self.d_scale * distances) / math.pi
        elif self.kernel == 'logistic':
            d = 2 / (1 + np.exp(-self.d_scale * distances)) - 1
        else:
            raise NotImplemented()
        d = np.sum(d) / len(distances)
        return self.max_reward - (self.max_reward - self.min_reward) * d

    def reset(self):
        self.step_count = 0
        self.previous_reward = 0.

    @abc.abstractmethod
    def get_reward(self, observation_point):
        pass
