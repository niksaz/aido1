import abc
import gc
import json
import time

import numpy as np
import requests
import torch.multiprocessing as torch_mp
from requests.exceptions import RequestException
from duckietown_rl.env import launch_env
from utils.reward_shaping.env_utils import Rewarder, Transformer, TargetTransformer
from utils.util import cut_off_leg

MAX_ITER = 1000


def create_env(config, internal_env_args, transfer):
    env = EnvironmentWrapper(config, internal_env_args, transfer)
    return env


class BaseEnvironment(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, action, project):
        pass

    @abc.abstractmethod
    def reset(self, project):
        pass

    @abc.abstractmethod
    def change_model(self, seed):
        pass

    @abc.abstractmethod
    def collect_garbage(self):
        pass

    @abc.abstractclassmethod
    def get_observation(self):
        pass

    def _create_env_from_type(self, env_init_args, internal_env_type):
        gc.collect()

        if internal_env_type == 'normal':
            self.env = DuckietownEnvironmentWrapper(**env_init_args)
        elif internal_env_type == 'virtual':
            self.env = VirtualEnvironment(**env_init_args)


class VirtualEnvironment(BaseEnvironment):
    def __init__(self, host_tcp, port_tcp):
        self.host_tcp = host_tcp
        self.port_tcp = port_tcp
        self.observation = None

    @staticmethod
    def _make_request(request, json_data={}):
        flag = True
        res = None
        while flag:
            try:
                res = requests.post(request, json=json_data).json()
            except RequestException:
                time.sleep(1)
                continue
            flag = False
        return res

    def step(self, action, project):
        action = action.tolist()
        json_data = json.dumps({'action': action, 'project': project})
        res = self._make_request('http://{host}:{port}/post_step_request/'.format(host=self.host_tcp,
                                                                                  port=self.port_tcp), json_data)

        self.observation = res['observation']
        return res['observation'], res['reward'], res['done'], res['info']

    def get_observation(self):
        return self.observation

    def reset(self, project):
        json_data = json.dumps({'project': project})
        res = self._make_request('http://{host}:{port}/post_reset_request/'.format(host=self.host_tcp,
                                                                                   port=self.port_tcp), json_data)

        self.observation = res['observation']
        return res['observation']

    def change_model(self, model, prosthetic, difficulty, seed, max_steps=None):
        if max_steps is None:
            json_data = json.dumps({'model': model, 'prosthetic': prosthetic, 'difficulty': difficulty, 'seed': seed})
        else:
            json_data = json.dumps({'model': model, 'prosthetic': prosthetic, 'difficulty': difficulty, 'seed': seed,
                                    'max_steps': max_steps})
        res = self._make_request('http://{host}:{port}/post_change_model_request/'.format(host=self.host_tcp,
                                                                                          port=self.port_tcp),
                                 json_data)

    def collect_garbage(self):
        res = self._make_request('http://{host}:{port}/post_collect_garbage_request/'.format(host=self.host_tcp,
                                                                                             port=self.port_tcp))


class DuckietownEnvironmentWrapper(BaseEnvironment):
    def __init__(self, visualize, integrator_accuracy=5e-5):
        self.visualize = visualize
        self.env = launch_env()
        self.observation = None
        self.seed = None

    def step(self, action, project):
        result = self.env.step(action, project)
        self.observation = result[0]
        return result

    def reset(self, project):
        self.observation = self.env.reset(project)
        return self.observation

    def get_observation(self):
        return self.observation

    def change_model(self, seed):
        if self.seed is None:
            self.seed = seed

        self.env.change_model(seed)

        return "ok"

    def collect_garbage(self):
        self.env = launch_env()
        self.change_model(self.seed)


class EnvironmentWrapper(BaseEnvironment):
    def __init__(self, config, internal_env_args, transfer):
        self.config = config
        self.env_config = self.config['environment']
        self.internal_env_type = internal_env_args['env_type']
        self.internal_env_init_args = internal_env_args['env_init_args']
        self.internal_env_config = internal_env_args['env_config']
        self.seed = None

        self.transfer = transfer

        self.env = None
        self._create_env_from_type(self.internal_env_init_args, self.internal_env_type)
        self.env.change_model(**self.internal_env_config)

        observation = self.env.reset(project=False)
        if observation == 'restart':
            raise ValueError('resetting timeout')

        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.env_step = 0
        self.total_reward = None

        self.observations.append(observation)

        self.wrapper_config = self.env_config['wrapper']

        self.repeat_actions = self.wrapper_config['repeat_actions']
        self.reward_scale = self.wrapper_config['reward_scale']
        self.fail_reward = self.wrapper_config['fail_reward']

        self.change_model(**self.internal_env_config)

        self.transformer = Transformer()
        self.rf_rewarder = Rewarder(self.wrapper_config['reward_functions'],
                                    self.wrapper_config['reward_aggregations'], observation)
        self.transformer.reset(observation)
        self.rf_rewarder.reset(observation)

        self.observation_transformed = self.transformer.transform(observation)
        self.submit_first_observation = self.transformer.transform(observation)

    def reset(self, project):
        observation = self.env.reset(project)

        if observation is None:
            return observation

        if observation == 'restart':
            raise ValueError('resetting timeout')

        self.observations = [observation]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.env_step = 0

        self.total_reward = 0.
        self.transformer = Transformer()
        self.transformer.reset(observation)
        self.rf_rewarder.reset(observation)
        self.observation_transformed = self.transformer.transform(observation)

        return self.observation_transformed, observation['body_pos']

    def collect_garbage(self):
        self.env.collect_garbage()

    def change_model(self, seed):
        if self.seed is None:
            self.env.change_model(seed)
            self.seed = seed

    def step(self, action, project=False):
        if self.config['model']['actor'][-1]['modules'][-1][-1]['name'] == 'tanh':
            action /= 2
            action += 0.5

        total_reward = 0.0
        total_reward_with_mod = 0.0

        if self.transfer:
            action = cut_off_leg(action)

        for _ in range(self.repeat_actions):
            observation, reward, done, _ = self.env.step(action, project=project)

            if observation == 'restart':
                raise ValueError('stepping timeout')

            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)

            modified_reward = self.rf_rewarder.reward(observation, None,
                                                      reward, action)

            total_reward += reward
            total_reward_with_mod += modified_reward

            self.env_step += 1

            if done:
                break

        self.observation_transformed = self.transformer.transform(observation)

        total_reward_with_mod *= self.reward_scale
        self.total_reward += total_reward

        return self.observation_transformed, (total_reward, total_reward_with_mod), done, None

    def get_observation(self):
        return self.observation_transformed

    @staticmethod
    def mirror_actions(actions):
        raise NotImplemented()

    def mirror_action(self, action):
        raise NotImplemented()


