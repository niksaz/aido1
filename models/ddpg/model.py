import abc
import json
import math
import random
from heapq import merge

from models.torch_utils import *
from utils.util import make_dir_if_required, parse_config
from ..ddpg.modules import Actor, Critic
from ..torch_utils import to_torch_tensor


class RealModel:
    @abc.abstractmethod
    def __init__(self, config):
        pass

    @abc.abstractmethod
    def act(self, observation, noise=0.0, cpu=False):
        pass

    '''
    PyTorch stuff
    '''

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def share_memory(self):
        pass

    @abc.abstractmethod
    def hard_update(self, source):
        pass

    @abc.abstractmethod
    def to(self, device):
        pass

    @abc.abstractmethod
    def init_critic(self, source):
        pass

    @abc.abstractmethod
    def init_actor(self, source):
        pass

    '''
    Serialization
    '''

    @abc.abstractmethod
    def save(self, config, directory, episode, reward):
        pass

    @abc.abstractmethod
    def load(self, directory):
        pass


class DDPG(RealModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.actor = Actor(config['actor'])
        self.critic = Critic(config['critic'])

        # self.actor = nn.DataParallel(self.actor)
        # self.critic = nn.DataParallel(self.critic)

    def act(self, observation, noise=0.0, cpu=False):
        observation = np.array([observation], dtype=np.float32)
        # print(observation[0, 0].max())
        # print(observation[0, 1].max())
        # print(observation[0, 2].max())
        # print(observation[0, 3].max())
        #
        # import matplotlib.pyplot as plt
        # f, axarr = plt.subplots(2, 2)
        # axarr[0, 0].imshow(observation[0, 0], cmap='gray')
        # axarr[0, 0].axis('off')
        # axarr[0, 1].imshow(observation[0, 1], cmap='gray')
        # axarr[0, 1].axis('off')
        # axarr[1, 0].imshow(observation[0, 2], cmap='gray')
        # axarr[1, 0].axis('off')
        # axarr[1, 1].imshow(observation[0, 3], cmap='gray')
        # axarr[1, 1].axis('off')
        # plt.show()

        with torch.no_grad():
            action = self.actor(
                to_torch_tensor(observation)
                # to_torch_tensor(observation_image_side)
                # to_torch_tensor(observation_image_front),
            ).cpu().numpy()

        action = np.squeeze(action)

        if self.config['actor'][-1]['modules'][-1][-1]['name'] == 'tanh':
            noise *= 2

        action += noise

        if self.config['actor'][-1]['modules'][-1][-1]['name'] == 'tanh':
            action = np.clip(action, -1.0, 1.0)
        if self.config['actor'][-1]['modules'][-1][-1]['name'] == 'sigmoid':
            action = np.clip(action, 0.0, 1.0)

        return action

    def train(self):
        self.actor.train()
        self.critic.train()

    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()

    def half(self):
        self.actor.half()
        self.critic.half()

    def hard_update(self, source):
        hard_update(self.actor, source.actor)
        hard_update(self.critic, source.critic)

    def get_actor(self):
        return self.actor

    def get_critic(self):
        return self.critic

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)

    def save(self, config, directory, episode, reward):
        make_dir_if_required(directory)
        episode_dir = '{}/episode_{}_reward_{:.2f}'.format(directory, episode, reward)
        make_dir_if_required(episode_dir)

        with open('{}/config.json'.format(episode_dir), 'w') as f:
            json.dump(config, f)

        self.save_weights(episode_dir)

    def save_weights(self, target_dir):
        torch.save(self.actor.state_dict(), '{}/actor_state_dict.pth'.format(target_dir))
        torch.save(self.critic.state_dict(), '{}/critic_state_dict.pth'.format(target_dir))

    def init_critic(self, source_ddpg):
        hard_update(self.critic, source_ddpg.critic)

    def init_actor(self, source_ddpg):
        hard_update(self.actor, source_ddpg.actor)

    def load(self, directory, load_gpu_model_on_cpu=False):
        if load_gpu_model_on_cpu:
            self.actor.load_state_dict(
                torch.load('{}/actor_state_dict.pth'.format(directory),
                           map_location=lambda storage, loc: storage))
            self.critic.load_state_dict(
                torch.load('{}/critic_state_dict.pth'.format(directory),
                           map_location=lambda storage, loc: storage))
        else:
            self.actor.load_state_dict(
                torch.load('{}/actor_state_dict.pth'.format(directory)))
            self.critic.load_state_dict(
                torch.load('{}/critic_state_dict.pth'.format(directory)))


class RemoteModel:
    def __init__(self, in_action_conn, out_observation_conn):
        self.in_action_conn = in_action_conn
        self.out_observation_conn = out_observation_conn

    def act(self, observation, noise=0.0, cpu=False):
        # print('remote model')
        # print(observation)
        # print()

        self.out_observation_conn.send((observation, noise))
        action = self.in_action_conn.recv()
        return action

    def save(self, config, directory, episode, reward):
        pass


class AverageModel:
    def __init__(self, models, config, repeats=1):
        self.models = models
        self.num_action = config["model"]["num_action"]
        self.repeats = repeats

    def act(self, observation, noise=0.0, cpu=False):
        action_sum = np.zeros(self.num_action, np.float32)
        for model in self.models:
            for _ in range(self.repeats):
                action_sum += model.act(observation, noise, cpu)
        return np.clip(action_sum / (len(self.models) * self.repeats), 0., 1.)

    def save(self, config, directory, episode, reward):
        pass


class ChooseRandomModel:
    def __init__(self, models):
        self.models = set(models)

    def act(self, observation, noise=0.0, cpu=False):
        return random.sample(self.models, 1)[0].act(observation, noise, cpu)

    def save(self, config, directory, episode, reward):
        pass


class MultiModelWrapper(RealModel):
    def __init__(self, config):
        super().__init__(config)
        self.angle_dividers = list(sorted(config["angle_dividers"]))
        self.speed_dividers = list(sorted(config["speed_dividers"]))
        self.step_dividers = list(sorted(config["step_dividers"]))
        self._total_models = get_total_indices(self.angle_dividers, self.speed_dividers, self.step_dividers)
        self.models = [DDPG(config) for _ in range(self._total_models)]

    def share_memory(self):
        for model in self.models:
            model.share_memory()

    def train(self):
        for model in self.models:
            model.train()

    def init_actor(self, source):
        if isinstance(source, MultiModelWrapper):
            for target_model, source_model in zip(self.models, source.models):
                target_model.init_actor(source_model)
        else:
            for model in self.models:
                model.init_actor(source)

    def init_critic(self, source):
        if isinstance(source, MultiModelWrapper):
            for target_model, source_model in zip(self.models, source.models):
                target_model.init_critic(source_model)
        else:
            for model in self.models:
                model.init_critic(source)

    def to(self, device):
        for model in self.models:
            model.to(device)

    def get_actors(self):
        return [model.get_actor() for model in self.models]

    def get_critics(self):
        return [model.get_critic() for model in self.models]

    def act(self, observation, noise=0.0, cpu=False):
        return self.models[self.find_index(observation[1])].act(observation[0], noise, cpu)

    def actions(self, observations):
        observations = map(lambda obs: (obs[0], self.find_index(obs[1])), observations)
        splited = [[] for _ in range(self._total_models)]
        for i, (obs, model_idx) in enumerate(observations):
            splited[model_idx].append((i, obs))
        actions = []
        for split, model in zip(splited, self.models):
            if len(split) == 0:
                continue
            idxs = [i for i, _ in split]
            obss = to_torch_tensor(np.array([obs for _, obs in split]))
            acts = zip(idxs, model.actor(obss).detach())
            actions.append(acts)
        actions = torch.stack([a[1] for a in merge(*actions, key=lambda t: t[0])])
        return actions

    def v_values(self, observations, actions):
        observations = map(lambda obs: (obs[0], self.find_index(obs[1])), observations)
        splited = [[] for _ in range(self._total_models)]
        for i, (obs, model_idx) in enumerate(observations):
            splited[model_idx].append((i, obs, actions[i]))
        values = []
        for split, model in zip(splited, self.models):
            if len(split) == 0:
                continue
            idxs = [i for i, _, _ in split]
            obss = to_torch_tensor(np.array([obs for _, obs, _ in split]))
            acts = torch.stack([act for _, _, act in split])
            v_vals = zip(idxs, model.critic(obss, acts).detach())
            values.append(v_vals)
        values = torch.stack([v[1] for v in merge(*values, key=lambda t: t[0])])
        return values

    def save(self, config, directory, episode, reward):
        make_dir_if_required(directory)
        episode_dir = '{}/episode_{}_reward_{:.2f}'.format(directory, episode, reward)
        make_dir_if_required(episode_dir)

        with open('{}/config.json'.format(episode_dir), 'w') as f:
            json.dump(config, f)

        for i in range(self._total_models):
            model_dir = '{}/{}'.format(episode_dir, i)
            make_dir_if_required(model_dir)
            self.models[i].save_weights(model_dir)

    def load(self, directory, load_gpu_model_on_cpu=False):
        for i in range(self._total_models):
            model_dir = '{}/{}'.format(directory, i)
            self.models[i].load(model_dir, load_gpu_model_on_cpu)

    def find_index(self, target_velocity):
        return find_index(target_velocity, self.angle_dividers, self.speed_dividers, self.step_dividers)

    def hard_update(self, source):
        for (target_model, source_model) in zip(self.models, source.models):
            target_model.hard_update(source_model)


def find_index(target, angle_dividers, speed_dividers, step_dividers):
    angle = math.atan2(target[0][2], target[0][0])
    speed = np.linalg.norm(target[0])
    step = target[1]
    angle_position = 0
    speed_position = 0
    step_position = 0
    result = 0
    while angle_position < len(angle_dividers) and angle > angle_dividers[angle_position]:
        angle_position += 1
    result = result * (len(angle_dividers) + 1) + angle_position
    while speed_position < len(speed_dividers) and speed > speed_dividers[speed_position]:
        speed_position += 1
    result = result * (len(speed_dividers) + 1) + speed_position
    while step_position < len(step_dividers) and step > step_dividers[step_position]:
        step_position += 1
    result = result * (len(step_dividers) + 1) + step_position
    return result


def get_total_indices(angle_dividers, speed_dividers, step_dividers):
    return (len(angle_dividers) + 1) * (len(speed_dividers) + 1) * (len(step_dividers) + 1)


def create_model(model_config):
    if model_config.get("enable_multimodel", False):
        return MultiModelWrapper(model_config)
    return DDPG(model_config)


def load_model(directory, load_gpu_model_on_cpu=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = create_model(parse_config(directory)['model'])
    model.load(directory, load_gpu_model_on_cpu)
    model.train()
    model.to(device)
    return model


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    directory = '.'


    model = create_model(parse_config(directory)['model'])
    model.train()
    model.to(device)

    input_to_net = torch.rand(4, 32, 64).numpy()

    with torch.no_grad():
        out = model.act(input_to_net)

    assert out.shape[0] == 2
