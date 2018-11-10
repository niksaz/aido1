import abc
import pickle
import random
import time

import numpy as np
import torch

from models.ddpg.model import create_model
from models.torch_utils import hard_update_ddpg
from utils.env_wrappers import create_env
from utils.logger import Logger
from utils.random_process import create_action_random_process
from utils.util import set_seeds, make_dir_if_required, create_decay_fn

MAGIC_NUMBER = 139


def explore_single_thread(exploration_type, config, p_id, model, internal_env_args,
                          queues,
                          best_reward, global_episode, global_update_step):

    explorer = SingleThreadExplorer(exploration_type, config, p_id, model, internal_env_args,
                                    queues,
                                    best_reward, global_episode, global_update_step)
    explorer.explore()


class Explorer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def explore(self):
        pass


class SingleThreadExplorer(Explorer):
    def __init__(self, exploration_type, config, p_id, model, internal_env_args,
                 episodes_queues, best_reward, global_episode, global_update_step):

        self.exploration_type = exploration_type
        self.config = config
        self.p_id = p_id

        if exploration_type == 'exploiting' or exploration_type == 'exploiting_virtual':
            self.model = create_model(config['model'])
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.model.train()
            self.model.to(device)
            self.training_model = model
        else:
            self.model = model

        self.explorer_seed = config['training']['global_seed'] + MAGIC_NUMBER * self.p_id
        self.explore_after = config['training'].get('explore_after', 0)
        self.steps_per_action = config['environment']['wrapper']['repeat_actions']
        set_seeds(self.explorer_seed)

        internal_env_args['env_config']['seed'] = self.explorer_seed

        self.environment = create_env(self.config, internal_env_args,
                                      transfer=config['training']['transfer'])

        self.episodes_queues = episodes_queues
        self.best_reward = best_reward
        self.saving_best_reward = -np.inf
        self.saving_reward_tolerance = None
        self.global_episode = global_episode
        self.global_update_step = global_update_step
        self.start_time = None
        self.logger = None

    def explore(self):
        training_config = self.config['training']
        self.saving_reward_tolerance = training_config['saving_reward_tolerance']

        dir_pattern = '{}' + '/{}_thread_{}'.format(self.exploration_type, self.p_id)

        logdir = dir_pattern.format(training_config['log_dir'])
        replays_dir = dir_pattern.format(training_config['replays_dir'])

        make_dir_if_required(logdir)
        if training_config['saving_replays']:
            make_dir_if_required(replays_dir)

        self.logger = Logger(logdir)

        episode_counter = 0
        step_counter = 0
        self.start_time = time.time()

        action_random_process = create_action_random_process(self.config)

        epsilon_cycle_len = random.randint(training_config['epsilon_cycle_len'] // 2,
                                           training_config['epsilon_cycle_len'] * 2)

        epsilon_decay_fn = create_decay_fn(
            "cycle",
            initial_value=training_config['initial_epsilon'],
            final_value=training_config['final_epsilon'],
            cycle_len=epsilon_cycle_len,
            num_cycles=training_config['max_episodes'] // epsilon_cycle_len)

        while True:
            try:
                if self.exploration_type == 'exploiting' or self.exploration_type == 'exploiting_virtual':
                    hard_update_ddpg(self.model, self.training_model)

                if episode_counter > 0 and episode_counter % 128 == 0:
                    self.environment.collect_garbage()

                epsilon = min(training_config['initial_epsilon'],
                              max(training_config['final_epsilon'], epsilon_decay_fn(episode_counter)))

                if self.exploration_type == 'exploiting' or self.exploration_type == 'exploiting_virtual':
                    epsilon = 0.0
                    self.explorer_seed = training_config['global_seed'] + MAGIC_NUMBER * self.p_id + episode_counter % 5
                    set_seeds(self.explorer_seed)

                episode_metrics = {
                    "reward": 0.0,
                    "reward_modified": 0.0,
                    "step": 0,
                    "epsilon": epsilon
                }

                action_random_process.reset_states()

                replay, timings = self._explore_episode(action_random_process, epsilon, episode_metrics, training_config)

                for episode_queue in self.episodes_queues:
                    episode_queue.put(replay)

                self.global_episode.value += 1

                episode_counter += 1
                episode_metrics["step"] *= self.config['environment']['wrapper']['repeat_actions']
                step_counter += episode_metrics["step"]

                reward_scale = self.config['environment']['wrapper']['reward_scale']
                episode_metrics['reward'] /= reward_scale
                episode_metrics['reward_modified'] /= reward_scale

                saving_best_cond = episode_metrics['reward'] > (self.saving_best_reward + self.saving_reward_tolerance)

                if saving_best_cond:
                    self.saving_best_reward = episode_metrics['reward']

                if (episode_counter % self.config['training']['save_every_episode'] == 0 or saving_best_cond) and \
                        (self.exploration_type == 'exploiting' or self.exploration_type == 'exploiting_virtual'):

                    save_dir = dir_pattern.format(self.config['training']['save_dir'])

                    self.model.save(self.config, save_dir, episode_counter, episode_metrics["reward"])

                self.log(episode_metrics, timings, episode_counter, step_counter, saving_best_cond)

            except ValueError as e:
                print('timedout process {} {} with {}'.format(self.exploration_type, self.p_id, e))

    @staticmethod
    def _save_replay(replay, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(replay, f)

    def _explore_episode(self, action_random_process, epsilon, episode_metrics, training_config):
        episode_timings = {'reset': time.time()}
        observation = self.environment.reset()
        episode_timings['reset'] = time.time() - episode_timings['reset']

        done = False
        replay = []
        action_random_process.reset_states()

        episode_timings['model'] = 0.0
        episode_timings['env'] = 0.0
        steps = 0

        while not done:
            sampled_action_noise = action_random_process.sample()

            observation_transformed = observation
            
            model_start = time.time()
            if self.exploration_type == 'exploiting' or self.exploration_type == 'exploiting_virtual':
                action = self.model.act(
                    observation_transformed
                )
            elif self.config['training']['every_second_random'] and self.p_id % 2 == 0 \
                    and random.uniform(0.0, 1.0) < self.config['training']['epsilon_ratio'] * epsilon:
                action = np.random.random(self.config['model']['num_action']).astype(dtype=np.float32)
            else:
                action = self.model.act(
                    observation_transformed,
                    noise=epsilon * sampled_action_noise
                )
            episode_timings['model'] += (time.time() - model_start)

            env_start = time.time()
            next_observation, (reward, reward_modified), done, _ = self.environment.step(action)
            steps += self.steps_per_action
            episode_timings['env'] += (time.time() - env_start)

            episode_metrics["reward"] += reward
            episode_metrics["reward_modified"] += reward_modified
            episode_metrics["step"] += 1

            if training_config['reward_modified']:
                reward = reward_modified

            replay.append((observation[0], action, reward, next_observation[0], done))

            observation = next_observation
            
        return replay, episode_timings

    def log(self, episode_metrics, episode_timings, episode_counter, step_counter, saving_best_cond):

        if episode_metrics["reward"] > self.best_reward.value:
            self.best_reward.value = episode_metrics["reward"]

        if saving_best_cond:
            self.logger.scalar_summary("best reward", self.saving_best_reward, episode_counter)

        elapsed_time = time.time() - self.start_time

        for key, value in episode_metrics.items():
            self.logger.scalar_summary(key, value, episode_counter)

        # self.logger.scalar_summary('reset env time, s', episode_timings['reset'], episode_counter)
        # self.logger.scalar_summary('model time, s', episode_timings['model'], episode_counter)
        # self.logger.scalar_summary('env step time', episode_timings['env'], episode_counter)

        self.logger.scalar_summary(
            "episode per minute",
            episode_counter / elapsed_time * 60,
            episode_counter)

        self.logger.scalar_summary(
            "step per second",
            step_counter / elapsed_time,
            episode_counter)
