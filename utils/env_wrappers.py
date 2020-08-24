import abc
import gc
import json
import time

import requests
from requests.exceptions import RequestException

from utils.env import launch_env
from utils.reward_shaping.env_utils import Rewarder, Transformer, \
    PreliminaryTransformer
from utils.util import from_numpy

MAX_ITER = 1000


def create_env(config, internal_env_args, transfer):
    env = EnvironmentWrapper(config, internal_env_args, transfer)
    return env


class BaseEnvironment:  # (metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, action):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def change_model(self, seed):
        pass

    @abc.abstractmethod
    def collect_garbage(self):
        pass

    @abc.abstractmethod
    def get_observation(self):
        pass

    def _create_env_from_type(self, env_init_args, internal_env_type, env_config=None):
        gc.collect()

        if internal_env_type == 'normal':
            self.env = DuckietownEnvironmentWrapper(**env_init_args, env_config=env_config)
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

    def step(self, action):
        print("{}: sending action".format(self.port_tcp))
        action = action.tolist()
        json_data = json.dumps({'action': action})
        res = self._make_request('http://{host}:{port}/post_step_request/'.format(host=self.host_tcp,
                                                                                  port=self.port_tcp), json_data)
        print("{}: observation received".format(self.port_tcp))
        self.observation = res['observation']
        return res['observation'], res['reward'], res['done'], res['info']

    def get_observation(self):
        return self.observation

    def reset(self):
        print("{}: reseting env".format(self.port_tcp))
        res = self._make_request('http://{host}:{port}/post_reset_request/'.format(host=self.host_tcp,
                                                                                   port=self.port_tcp))

        print("{}: env reseted".format(self.port_tcp))
        self.observation = res['observation']
        return res['observation']

    def change_model(self, seed):
        json_data = json.dumps({'seed': seed})
        res = self._make_request('http://{host}:{port}/post_change_model_request/'.format(host=self.host_tcp,
                                                                                          port=self.port_tcp),
                                 json_data)

    def collect_garbage(self):
        res = self._make_request('http://{host}:{port}/post_collect_garbage_request/'.format(host=self.host_tcp,                                                                              port=self.port_tcp))


class DuckietownEnvironmentWrapper(BaseEnvironment):
    def __init__(self, name=None, env_config=None):
        self.env = launch_env(name)
        self.observation = None
        self.seed = None
        preliminary_transformer_kwargs = env_config['wrapper'].get('preliminary_transformer', {})
        self.preliminary_transformer = PreliminaryTransformer(**preliminary_transformer_kwargs)

    def step(self, action):
        result = list(self.env.step(action))
        result[0] = self.preliminary_transformer.transform(result[0])
        result = [from_numpy(data) for data in result]
        self.observation = result[0]
        return result

    def reset(self):
        obs = self.env.reset()

        self.preliminary_transformer.reset(obs)
        self.observation = from_numpy(self.preliminary_transformer.transform(obs))
        return self.observation

    def get_observation(self):
        return self.observation

    def change_model(self, seed):
        self.seed = seed
        self.env.seed(seed)

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
        self._create_env_from_type(self.internal_env_init_args, self.internal_env_type, self.env_config)
        self.env.change_model(**self.internal_env_config)

        observation = self.env.reset()
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

        self.max_env_steps = self.wrapper_config['max_env_steps']
        self.repeat_actions = self.wrapper_config['repeat_actions']
        self.reward_scale = self.wrapper_config['reward_scale']

        self.change_model(**self.internal_env_config)

        self.transformer = Transformer()
        self.rf_rewarder = Rewarder(self.wrapper_config['reward_functions'],
                                    self.wrapper_config['reward_aggregations'], observation)
        self.transformer.reset(observation)
        self.rf_rewarder.reset(observation)

        self.observation_transformed = self.transformer.transform(observation)
        self.submit_first_observation = self.transformer.transform(observation)

    def reset(self):
        observation = self.env.reset()

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

        return self.observation_transformed

    def collect_garbage(self):
        self.env.collect_garbage()

    def change_model(self, seed):
        if self.seed is None:
            self.env.change_model(seed)
            self.seed = seed

    def step(self, action):
        total_reward = 0.0
        total_reward_with_mod = 0.0

        for _ in range(self.repeat_actions):
            observation, reward, done, info = self.env.step(action)

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

            done = done or self.env_step > self.max_env_steps

            if done:
                break

        self.observation_transformed = self.transformer.transform(observation)

        total_reward_with_mod *= self.reward_scale
        self.total_reward += total_reward

        return self.observation_transformed, (total_reward, total_reward_with_mod), done, info

    def get_observation(self):
        return self.observation_transformed

    @staticmethod
    def mirror_actions(actions):
        raise NotImplemented()

    def mirror_action(self, action):
        raise NotImplemented()


