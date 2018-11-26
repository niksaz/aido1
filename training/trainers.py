import abc
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.adamw import AdamW
from models.torch_utils import soft_update, to_torch_tensor, average_update, hard_update
from utils.logger import Logger
from utils.util import set_seeds, make_dir_if_required, TrainingDecay

MAGIC_NUMBER = 179


def weighted_mse_loss(input, target, weights):
    out = (input - target) ** 2
    loss = out.mean(0)
    return loss


def train_single_thread(config, p_id, target_model, proxy_model, models, start_barrier, finish_barrier, update_lock,
                        queue, best_reward, global_episode, global_update_step):
    if config['model'].get("enable_multimodel", False) and \
            not config["training"].get("multimodel_parallelism_enabled", False):
        trainer = MultiDDPGTrainer(config, p_id, target_model, proxy_model, models, start_barrier, finish_barrier,
                                   update_lock, queue, best_reward, global_episode, global_update_step)
    else:
        trainer = DDPGTrainer(config, p_id, target_model, proxy_model, models, start_barrier, finish_barrier,
                              update_lock, queue, best_reward, global_episode, global_update_step)
    trainer.train()


class BaseTrainer:
    def __init__(self, config, p_id, target_model, proxy_model, models, start_barrier, finish_barrier, update_lock,
                 sample_queue, best_reward, global_episode, global_update_step):
        self.config = config
        self.p_id = p_id
        self.target_model = target_model

        self.start_barrier = start_barrier
        self.finish_barrier = finish_barrier
        self.update_lock = update_lock
        self.sample_queue = sample_queue
        self.best_reward = best_reward
        self.global_episode = global_episode
        self.global_update_step = global_update_step
        self.global_update_step_with_weight_averaging = 0
        self.logger = None
        self.start_time = None
        self.actor_decay = None
        self.critic_decay = None
        self.criterion = None

    @abc.abstractmethod
    def update(self, train_data):
        return None, None

    @abc.abstractmethod
    def average_model(self):
        pass

    @abc.abstractmethod
    def update_model_by_proxy(self):
        pass

    @abc.abstractmethod
    def update_target_model(self):
        pass

    def train(self):
        training_config = self.config['training']

        trainer_seed = training_config['global_seed'] + MAGIC_NUMBER * self.p_id
        set_seeds(trainer_seed)

        trainer_logdir = '{}/train_thread_{}'.format(training_config["log_dir"], self.p_id)
        make_dir_if_required(trainer_logdir)
        self.logger = Logger(trainer_logdir)

        self.start_time = time.time()

        update_step = 0
        received_examples = 1  # just hack
        buffer_size = 0

        self.criterion = torch.nn.MSELoss()

        self.actor_decay = TrainingDecay(training_config['actor_train_decay'])
        self.critic_decay = TrainingDecay(training_config['critic_train_decay'])

        while True:
            #critic_lr = self.critic_lr_decay_fn(self.global_update_step.value)
            #actor_lr = self.actor_lr_decay_fn(self.global_update_step.value)

            if update_step > 0:
                train_data, received_examples, buffer_size = self.sample_queue.get()

                step_metrics, step_info = self.update(train_data)

                for key, value in step_metrics.items():
                    self.logger.scalar_summary(key, value, update_step)

                #self.logger.scalar_summary("actor lr", actor_lr, update_step)
                #self.logger.scalar_summary("critic lr", critic_lr, update_step)
            else:
                time.sleep(training_config['train_delay'])

            update_step += 1
            print('Trainer:train -> update step', update_step)

            self.logger.scalar_summary("buffer size", buffer_size, self.global_episode.value)
            self.logger.scalar_summary(
                "updates per example",
                update_step * training_config['batch_size'] / received_examples,
                self.global_episode.value)
            self.logger.scalar_summary(
                "updates per example global",
                self.global_update_step.value * training_config['batch_size'] / received_examples,
                self.global_episode.value)


class DDPGTrainer(BaseTrainer):
    def __init__(self, config, p_id, target_model, proxy_model, models, start_barrier, finish_barrier, update_lock,
                 sample_queue, best_reward, global_episode, global_update_step):
        super().__init__(config, p_id, target_model, proxy_model, models, start_barrier, finish_barrier, update_lock,
                         sample_queue, best_reward, global_episode, global_update_step)
        self.proxy_model = proxy_model
        self.proxy_actor = self.proxy_model.get_actor()
        self.proxy_critic = self.proxy_model.get_critic()

        self.models = models
        self.model = self.models[p_id]
        self.actors = [model.get_actor() for model in self.models]
        self.critics = [model.get_critic() for model in self.models]

        self.target_actor = None
        self.target_critic = None
        self.actor = None
        self.critic = None
        self.actor_optim = None
        self.critic_optim = None

    def update(self, train_data):
        observations, actions, rewards, next_observations, dones = train_data
        dones = dones[:, None].astype(np.bool)
        rewards = rewards[:, None].astype(np.float32)

        dones = to_torch_tensor(np.invert(dones).astype(np.float32))
        rewards = to_torch_tensor(rewards)

        self.critic_decay.update_step(self.global_update_step.value)
        self.actor_decay.update_step(self.global_update_step.value)

        # Critic update

        next_actions = self.target_actor(
            to_torch_tensor(next_observations)
        ).detach()

        next_v_values = self.target_critic(
            to_torch_tensor(next_observations),
            next_actions
        ).detach()

        y_expected = rewards + dones * self.config['training']['gamma'] * next_v_values

        y_predicted = self.critic(
            to_torch_tensor(observations),
            to_torch_tensor(actions)
        )

        if self.config['training']['critic_loss'] == 'mse_loss':
            critic_loss = F.mse_loss(y_predicted, y_expected)
        if self.config['training']['critic_loss'] == 'smooth_l1_loss':
            critic_loss = F.smooth_l1_loss(y_predicted, y_expected)

        self.critic_optim.zero_grad()
        critic_loss.backward()

        for param_group in self.critic_optim.param_groups:
            self.critic_decay(param_group)

        self.critic_optim.step()

        # Actor update

        pred_actions = self.actor(
            to_torch_tensor(observations)
        )

        actor_loss = self.critic(
            to_torch_tensor(observations),
            pred_actions
        )

        actor_loss = -1.0 * torch.mean(actor_loss)

        self.actor_optim.zero_grad()
        actor_loss.backward()

        for param_group in self.actor_optim.param_groups:
            self.actor_decay(param_group)

        self.actor_optim.step()

        if self.config['training']['num_threads_training'] > 1 \
                and (self.global_update_step_with_weight_averaging +
                         self.config['training']['update_steps_between_update']) < self.global_update_step.value:
            self.start_barrier.wait()
            self.average_model()
            self.global_update_step_with_weight_averaging = self.global_update_step.value
            self.finish_barrier.wait()
            self.update_model_by_proxy()

        with self.update_lock:
            self.update_target_model()

        metrics = {
            "critic_loss": np.sqrt(critic_loss.item()),
            "actor_loss": actor_loss.item()
        }

        with torch.no_grad():
            td_v_values = self.critic(to_torch_tensor(observations),
                                      to_torch_tensor(actions)
                                      )

            td_error = y_expected - td_v_values
            td_error = td_error.cpu().numpy()

        info = {
            'td_error': td_error
        }

        self.global_update_step.value += 1

        return metrics, info

    def average_model(self):
        average_update(self.proxy_actor, self.actors)
        average_update(self.proxy_critic, self.critics)

    def update_model_by_proxy(self):
        hard_update(self.actor, self.proxy_actor)
        hard_update(self.critic, self.proxy_critic)

    def update_target_model(self):
        soft_update(self.target_actor, self.actor, self.config['training']['tau'])
        soft_update(self.target_critic, self.critic, self.config['training']['tau'])

    def train(self):
        training_config = self.config['training']

        self.target_actor = self.target_model.get_actor()
        self.target_critic = self.target_model.get_critic()
        self.actor = self.model.get_actor()
        self.critic = self.model.get_critic()

        if training_config["optimizer"] == "adam":
            self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.)
            self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.)
        elif training_config["optimizer"] == "adamw":
            self.actor_optim = AdamW(self.actor.parameters(), lr=0., weight_decay=0.0001)
            self.critic_optim = AdamW(self.critic.parameters(), lr=0., weight_decay=0.0001)
        elif training_config["optimizer"] == "sgd":
            self.actor_optim = torch.optim.SGD(self.actor.parameters(), lr=0., momentum=0.9, weight_decay=0.0001)
            self.critic_optim = torch.optim.SGD(self.critic.parameters(), lr=0., momentum=0.9, weight_decay=0.0001)
        else:
            raise NotImplemented()
        super().train()


class MultiDDPGTrainer(BaseTrainer):
    def __init__(self, config, p_id, target_model, proxy_model, models, start_barrier, finish_barrier, update_lock,
                 sample_queue, best_reward, global_episode, global_update_step):
        super().__init__(config, p_id, target_model, proxy_model, models, start_barrier, finish_barrier, update_lock,
                         sample_queue, best_reward, global_episode, global_update_step)
        self.proxy_model = proxy_model

        self.models = models
        self.model = self.models[p_id]

        self.actor_optims = None
        self.critic_optims = None

    def train(self):
        training_config = self.config['training']

        if training_config["optimizer"] == "adam":
            self.actor_optims = [torch.optim.Adam(actor.parameters(), lr=0.) for actor in self.model.get_actors()]
            self.critic_optims = [torch.optim.Adam(critic.parameters(), lr=0.) for critic in self.model.get_critics()]
        elif training_config["optimizer"] == "adamw":
            self.actor_optims = [AdamW(actor.parameters()) for actor in self.model.get_actors()]
            self.critic_optims = [AdamW(critic.parameters()) for critic in self.model.get_critics()]
        elif training_config["optimizer"] == "sgd":
            self.actor_optims = [torch.optim.SGD(actor.parameters(), lr=0., momentum=0.9, weight_decay=0.0001)
                                for actor in self.model.get_actors()]
            self.critic_optims = [torch.optim.SGD(critic.parameters(), lr=0., momentum=0.9, weight_decay=0.0001)
                                 for critic in self.model.get_critics()]
        else:
            raise NotImplemented()
        super().train()

    def update_model_by_proxy(self):
        for i in range(len(self.model.models)):
            hard_update(self.model.get_actors()[i], self.proxy_model.get_actors()[i])
            hard_update(self.model.get_critics()[i], self.proxy_model.get_critics()[i])

    def update_target_model(self):
        for i in range(len(self.model.models)):
            soft_update(self.target_model.get_actors()[i], self.model.get_actors()[i], self.config['training']['tau'])
            soft_update(self.target_model.get_critics()[i], self.model.get_critics()[i], self.config['training']['tau'])

    def average_model(self):
        for i in range(len(self.model.models)):
            average_update(self.proxy_model.get_actors()[i], [model.get_actors()[i] for model in self.models])
            average_update(self.proxy_model.get_critics()[i], [model.get_critics()[i] for model in self.models])

    def update(self, train_data):
        self.critic_decay.update_step(self.global_update_step.value)
        self.actor_decay.update_step(self.global_update_step.value)

        avg_critic_loss = 0.
        avg_actor_loss = 0.

        for i, (observations, actions, rewards, next_observations, dones) in enumerate(train_data):
            dones = dones[:, None].astype(np.bool)
            rewards = rewards[:, None].astype(np.float32)

            dones = to_torch_tensor(np.invert(dones).astype(np.float32))
            rewards = to_torch_tensor(rewards)

            # Critic update

            next_actions = self.target_model.actions(next_observations)
            next_v_values = self.target_model.v_values(next_observations, next_actions)
            observations_tensor = to_torch_tensor(np.array([observation[0] for observation in observations]))

            y_expected = rewards + dones * self.config['training']['gamma'] * next_v_values

            y_predicted = self.model.get_critics()[i](observations_tensor, to_torch_tensor(actions))

            if self.config['training']['critic_loss'] == 'mse_loss':
                critic_loss = F.mse_loss(y_predicted, y_expected)
            if self.config['training']['critic_loss'] == 'smooth_l1_loss':
                critic_loss = F.smooth_l1_loss(y_predicted, y_expected)

            self.critic_optims[i].zero_grad()
            critic_loss.backward()

            for param_group in self.critic_optims[i].param_groups:
                self.critic_decay(param_group)

            self.critic_optims[i].step()
            avg_critic_loss += critic_loss.item()

            # Actor update

            pred_actions = self.model.get_actors()[i](observations_tensor)

            actor_loss = self.model.get_critics()[i](observations_tensor, pred_actions)

            actor_loss = -1.0 * torch.mean(actor_loss)

            self.actor_optims[i].zero_grad()
            actor_loss.backward()

            for param_group in self.actor_optims[i].param_groups:
                self.actor_decay(param_group)

            self.actor_optims[i].step()
            avg_actor_loss += actor_loss.item()

        avg_critic_loss /= len(train_data)
        avg_actor_loss /= len(train_data)

        if self.config['training']['num_threads_training'] > 1 \
                and (self.global_update_step_with_weight_averaging +
                         self.config['training']['update_steps_between_update']) < self.global_update_step.value:
            self.start_barrier.wait()
            self.average_model()
            self.global_update_step_with_weight_averaging = self.global_update_step.value
            self.finish_barrier.wait()
            self.update_model_by_proxy()

        with self.update_lock:
            self.update_target_model()

        metrics = {
            "critic_loss": np.sqrt(avg_critic_loss),
            "actor_loss": avg_actor_loss
        }

        info = {
            # 'td_error': td_error
        }

        self.global_update_step.value += 1

        return metrics, info
