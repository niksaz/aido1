import abc
import gc
import json
import time

import numpy as np
import requests
import torch.multiprocessing as torch_mp
from osim.env import ProstheticsEnv
from osim.http.client import Client
from requests.exceptions import RequestException

from utils.graphics import VirtualGraphics
from utils.math_utils import rotate_coors
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
    def change_model(self, model, prosthetic, difficulty, seed, max_steps=None):
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
            self.env = ProstheticEnvironmentWrapper(**env_init_args)
        elif internal_env_type == 'proxy':
            self.env = ProstheticEnvironmentProxyWrapper(**env_init_args)
        elif internal_env_type == 'virtual':
            self.env = VirtualEnvironment(**env_init_args)
        elif internal_env_type == 'submit':
            self.env = SubmitEnvironmentWrapper(**env_init_args)
        elif internal_env_type == 'visualizer':
            self.env = VisualizeEnvironmentWrapper(**env_init_args)


class VisualizeEnvironmentWrapper(BaseEnvironment):
    def get_observation(self):
        pass

    def __init__(self, env_init_args, env_type, visualizers_configs):
        self._create_env_from_type(env_init_args, env_type)
        self.visualizers_configs = visualizers_configs

    def collect_garbage(self):
        return self.env.collect_garbage()

    def step(self, action, project):
        self.step_counter += 1
        observation, reward, done, n = self.env.step(action, project)
        for vis in self.visualizers:
            vis.refresh_frame(observation, reward, self.step_counter)
        return observation, reward, done, n

    def change_model(self, **kwargs):
        return self.env.change_model(**kwargs)

    def reset(self, project):
        self._init_visualizers()
        observation = self.env.reset(project)
        for vis in self.visualizers:
            vis.refresh_frame(observation, 0., self.step_counter)
        return observation

    def _init_visualizers(self):
        self.step_counter = 0
        self.visualizers = [VirtualGraphics(config) for config in self.visualizers_configs]


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


def env_worker(in_command, out_result, integrator_accuracy=5e-5, max_steps=None):
    env = ProstheticEnvironmentWrapper(visualize=False, integrator_accuracy=integrator_accuracy, max_steps=max_steps)

    while True:
        command, kwargs = in_command.recv()
        if command == 'step':
            result = env.step(**kwargs)
        elif command == 'reset':
            result = env.reset(**kwargs)
        elif command == 'change_model':
            result = env.change_model(**kwargs)

        out_result.send(result)


class ProstheticEnvironmentProxyWrapper(BaseEnvironment):
    timeout = 180

    def __init__(self, visualize, integrator_accuracy=5e-5, max_steps=None):
        self.visualize = visualize
        self.integrator_accuracy = integrator_accuracy
        self.model = None
        self.prosthetic = None
        self.difficulty = None
        self.seed = None
        self.daemon_process = None
        self.in_command = None
        self.out_command = None
        self.in_result = None
        self.out_result = None
        self.observation = None
        self.max_steps = max_steps
        self._init_conns()
        self._start_daemon()

    def _init_conns(self):
        if self.in_command is not None:
            self.in_command.close()

        if self.out_command is not None:
            self.out_command.close()

        if self.in_result is not None:
            self.in_result.close()

        if self.out_result is not None:
            self.out_result.close()

        self.in_command, self.out_command = torch_mp.Pipe(duplex=False)
        self.in_result, self.out_result = torch_mp.Pipe(duplex=False)

    def _start_daemon(self):
        self.daemon_process = torch_mp.Process(
            target=env_worker,
            args=(self.in_command, self.out_result, self.integrator_accuracy, self.max_steps)
        )
        self.daemon_process.daemon = True
        self.daemon_process.start()

    def _perform_command(self, command, kwargs):
        self.out_command.send((command, kwargs))
        if self.in_result.poll(self.timeout):
            return self.in_result.recv()
        else:
            self._init_conns()
            self.collect_garbage()
            return 'restart'

    def step(self, action, project):
        result = self._perform_command('step', {'action': action, 'project': project})
        if result == 'restart':
            self.observation = None
            return 'restart', 'restart', 'restart', 'restart'
        else:
            self.observation = result[0]
            return result

    def reset(self, project):
        gc.collect()
        torch_mp.active_children()
        result = self._perform_command('reset', {'project': project})
        self.observation = result
        return result

    def get_observation(self):
        return self.observation

    def change_model(self, model, prosthetic, difficulty, seed, max_steps=None):
        if self.model is None:
            self.model = model
            self.prosthetic = prosthetic
            self.max_steps = max_steps
            self.difficulty = difficulty
            self.seed = seed
        self._perform_command('change_model', {'model': model, 'prosthetic': prosthetic,
                                               'difficulty': difficulty, 'seed': seed, 'max_steps': max_steps})

    def collect_garbage(self):
        self.daemon_process.terminate()
        torch_mp.active_children()
        self._start_daemon()
        self.change_model(self.model, self.prosthetic, self.difficulty, self.seed)


class ProstheticEnvironmentWrapper(BaseEnvironment):
    def __init__(self, visualize, integrator_accuracy=5e-5, max_steps=None):
        self.visualize = visualize
        self.time_limit = max_steps
        self.integrator_accuracy = integrator_accuracy
        self.env = ProstheticsEnv(visualize=self.visualize, integrator_accuracy=self.integrator_accuracy)
        if self.time_limit is not None:
            self.env.time_limit = self.time_limit
            self.env.spec.timestep_limit = self.time_limit
        self.model = None
        self.prosthetic = None
        self.difficulty = None
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

    def change_model(self, model, prosthetic, difficulty, seed, max_steps=None):
        if self.model is None:
            self.model = model
            self.prosthetic = prosthetic
            self.difficulty = difficulty
            if self.time_limit is None:
                self.time_limit = max_steps
            self.seed = seed

        self.env.change_model(model, prosthetic, difficulty, seed)
        if self.time_limit is not None:
            self.env.time_limit = self.time_limit
            self.env.spec.timestep_limit = self.time_limit

        return "ok"

    def collect_garbage(self):
        self.env = ProstheticsEnv(visualize=self.visualize)
        self.change_model(self.model, self.prosthetic, self.difficulty, self.seed)


class SubmitEnvironmentWrapper(BaseEnvironment):
    def __init__(self, test_speed):
        self.env = Client(EnvironmentWrapper.remote_base)
        self.counter = 0
        self.test_speed = test_speed
        self.observation = None

    def step(self, action, project):
        action = action.tolist()
        observation, reward, done, info = self.env.env_step(action, render=True)
        if self.test_speed:
            observation["target_vel"] = np.array([1.25, 0., 0.])
        self.observation = observation
        return observation, reward, done, info

    def reset(self, project):
        if self.counter == 0:
            observation = self.env.env_create(EnvironmentWrapper.crowdai_token, env_id='ProstheticsEnv')
        else:
            observation = self.env.env_reset()

        if observation is None:
            self.env.submit()
            return observation

        self.counter += 1

        if self.test_speed:
            observation["target_vel"] = np.array([1.25, 0., 0.])

        self.observation = observation

        return observation

    def get_observation(self):
        return self.observation

    def change_model(self, model, prosthetic, difficulty, seed, max_steps=None):
        pass

    def collect_garbage(self):
        pass


class EnvironmentWrapper(BaseEnvironment):
    remote_base = "http://grader.crowdai.org:1730"
    crowdai_token = "25ed6861f07169704b55127aad34c33a"

    def __init__(self, config, internal_env_args, transfer):
        self.config = config
        self.multimodel_enabled = self.config["model"].get("enable_multimodel", False)
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
        self.staying_alive_reward = self.wrapper_config['staying_alive_reward']

        self.target_transformer = TargetTransformer(self.wrapper_config.get('target_transformer_type', "normal"),
                                                    self.wrapper_config.get('target_transformer_config', {}))
        self.change_model(**self.internal_env_config)

        self.transformer = Transformer(self.wrapper_config['features'], observation,
                                       self.transfer, 100/self.repeat_actions,
                                       self.wrapper_config.get('target_noise', 0.02))
        self.pf_rewarder = Rewarder(self.wrapper_config['potential_functions'], {}, observation)
        self.rf_rewarder = Rewarder(self.wrapper_config['reward_functions'],
                                    self.wrapper_config['reward_aggregations'], observation)
        self.target_transformer.reset()
        self.transformer.reset(observation)
        self.pf_rewarder.reset(observation)
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

        self.target_transformer.reset(self.total_reward)
        self.total_reward = 0.
        self.transformer = Transformer(self.wrapper_config['features'], observation,
                                       self.transfer, 100/self.repeat_actions)
        self.transformer.reset(observation)
        self.pf_rewarder.reset(observation)
        self.rf_rewarder.reset(observation)
        self.observation_transformed = self.transformer.transform(observation)
        if self.multimodel_enabled:
            self.observation_transformed = (self.observation_transformed, (observation["target_vel"], 0))

        return self.observation_transformed, observation['body_pos']

    def collect_garbage(self):
        self.env.collect_garbage()

    def change_model(self, model, prosthetic, difficulty, seed, max_steps=None):
        if self.seed is None:
            self.seed = seed
            self.target_transformer.set_seed(seed)

    def step(self, action, project=False):
        if self.config['model']['actor'][-1]['modules'][-1][-1]['name'] == 'tanh':
            action /= 2
            action += 0.5

        total_reward = 0.0
        total_reward_with_mod = self.pf_rewarder.reward(self.observations[-1],
                                                        self.observation_transformed, 0., action)

        if self.transfer:
            action = cut_off_leg(action)

        for _ in range(self.repeat_actions):
            observation, reward, done, _ = self.env.step(action, project=project)

            if observation == 'restart':
                raise ValueError('stepping timeout')

            observation, reward = self.target_transformer.transform(observation, reward)

            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)

            modified_reward = self.rf_rewarder.reward(observation, None,
                                                      reward, action)

            total_reward += reward
            total_reward_with_mod += modified_reward

            self.env_step += 1

            if not done:
                total_reward_with_mod += self.staying_alive_reward
            else:
                if self.env_step < MAX_ITER:
                    total_reward_with_mod += self.fail_reward
                break

        if self.config['environment']['rotate_coors']:
            observation = rotate_coors(observation)
        self.observation_transformed = self.transformer.transform(observation)
        if self.multimodel_enabled:
            self.observation_transformed = (self.observation_transformed, (observation["target_vel"], self.env_step))

        obs_for_image = observation['body_pos']

        total_reward_with_mod *= self.reward_scale
        self.total_reward += total_reward

        return (self.observation_transformed, obs_for_image), (total_reward, total_reward_with_mod), done, None

    def get_raw_replay(self):
        return self.observations, self.actions, self.rewards, self.dones

    def mirror_legs(self, observation):
        return self.transformer.mirror_legs(observation)

    def get_observation(self):
        return self.observation_transformed

    def get_episode_info(self):
        result = []
        for observation, action, done, reward in zip(self.observations, self.actions, self.dones, self.rewards):
            result.append(
                {
                    "observation": observation,
                    "action": [float(a) for a in action],
                    "done": done,
                    "reward": reward
                }
            )
        return result

    @staticmethod
    def mirror_actions(actions):
        mod_actions = np.asarray(actions)
        num_actions = actions.shape[1]
        mod_actions[:, :num_actions // 2] = actions[:, num_actions // 2:]
        mod_actions[:, num_actions // 2:] = actions[:, :num_actions // 2]
        return mod_actions

    def mirror_action(self, action):
        action = np.reshape(action, (1, -1))
        action = self.mirror_actions(action)
        action = action.flatten()
        return action


