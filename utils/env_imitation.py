from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import List

import cv2
import numpy as np

import gym

@dataclass
class EpisodeStep:
    action: np.ndarray
    observation: np.ndarray

@dataclass
class Episode:
    steps: List[EpisodeStep]

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, i):
        return self.steps[i]


class ImitationEnvironment(gym.Env):
    def __init__(self, folder_path, sort_fn=None):
        """
        :param sort_fn: function to sort collected samples and return list of `Episode`
        """
        folder_path = Path(folder_path)
        self.episodes: List[Episode] = sort_fn(folder_path) if sort_fn else self.default_sort_fn(folder_path)
        self.current_episode = -1  # need to start with -1 because on the first reset will add 1
        self.current_step = 0
        self.errors = []

    def default_sort_fn(self, folder_path):
        """
        some assumptions to use this default function:
        all collected samples should be in `ep_{episode_number}_{step_number}` format
        also observation should be in `.png` and action in `.npy` extension
        """
        # TODO(ybelousov) update to lazy loading if needed (to not load all in memory at once)
        episodes = []
        # filter only needed files
        paths = [path for path in folder_path.iterdir() if path.stem.endswith(('.png', '.npy'))]
        # first sort by episode number, than by episode step, and than by file format (action before observation)
        sorted_directory = sorted(paths, key=lambda x: (int(x.stem.split('_')[1]), int(x.stem.split('_')[2]), x.suffix))
        for episode_num, episode_directory in groupby(sorted_directory, lambda x: int(x.stem.split('_')[1])):
            steps = []
            for action_filename, observation_filename in zip(episode_directory, episode_directory):
                action = np.load(action_filename)
                observation = cv2.cvtColor(cv2.imread(str(observation_filename)), cv2.COLOR_BGR2RGB)
                steps.append(EpisodeStep(action=action, observation=observation))
            episodes.append(Episode(steps=steps))
        return episodes

    def reset(self):
        assert self.current_episode + 1 < len(self.episodes), "All episodes passed"
        self.current_episode += 1
        self.current_step = 0
        self.errors.append([])
        return self.episodes[self.current_episode][self.current_step].observation

    def step(self, action):
        """
        should return observation, reward, done, info
        """
        assert self.current_step < len(self.episodes[self.current_episode]), "All steps in episode passed"
        error = np.linalg.norm(action - self.episodes[self.current_episode][self.current_step].action)
        self.errors[self.current_episode].append(error)
        self.current_step += 1
        done = (self.current_step >= len(self.episodes[self.current_episode]))
        observation = None if done else self.episodes[self.current_episode][self.current_step].observation
        return observation, error, done, {}


if __name__ == "__main__":
    logs_path = "collected_logs/samples-collect"
    env = ImitationEnvironment(folder_path=logs_path)

    obs = env.reset()

    done = False

    while not done:
        action = np.random.rand(2)
        obs, error, done, info = env.step(action)

    print(sum(env.errors[0]))
