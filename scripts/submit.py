import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import numpy as np

from models.ddpg.model import create_model, AverageModel, ChooseRandomModel
from utils.env_wrappers import create_env
from utils.util import set_seeds, parse_config, save_info


def tta(model, env, observation, mirrored_observation):
    action = model.act(observation)
    mirrored_action = model.act(mirrored_observation)
    unmirrored_action = env.mirror_action(mirrored_action)
    averaged_action = (action + unmirrored_action) / 2
    averaged_action = np.clip(averaged_action, 0.0, 1.0)
    return averaged_action


def submit(config, directories, repeats=1):
    explorer_seed = config['training']['global_seed'] + 0
    set_seeds(explorer_seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    avg_models = []

    for dirs in directories:
        models = []
        for model_directory in dirs:
            model = create_model(config['model'])
            model.load(model_directory)
            model.train()
            model.to(device)
            models.append(model)
        avg_models.append(AverageModel(models, config, repeats))

    model = ChooseRandomModel(avg_models)

    internal_env_args = {'env_type': 'submit',
                         'env_init_args': {'test_speed': False},
                         'env_config': {
                             "model": "3D",
                             "prosthetic": True,
                             "difficulty": 1,
                             'seed': explorer_seed}
                         }

    env = create_env(config, internal_env_args, config['training']['transfer'])

    observation = env.get_observation()

    episodes = 0
    counter = 0
    reward_sum = 0.0

    while True:
        action = model.act(observation)

        (observation, _), (reward, _), done, _ = env.step(action)
        counter += 1

        reward_sum += reward
        print(counter, reward, reward_sum)

        if done:
            print()
            counter = 0
            reward_sum = 0
            save_info("submit_logs/second/log_{}.json".format(episodes), env.get_episode_info())
            episodes += 1
            (observation, _) = env.reset(False)
            if observation is None:
                break


if __name__ == '__main__':
    directory = '/data/svidchenko/afterlearning/moar_features_1/sgdr_1/saved_models/exploiting_virtual_thread_0/episode_30_reward_9841.04/'

    directories = [
        [
            #'/data/svidchenko/afterlearning/moar_features_1/sgdr_1/saved_models/exploiting_virtual_thread_0/episode_48_reward_9840.57/',
            #'/data/svidchenko/afterlearning/moar_features_1/sgdr_1/saved_models/exploiting_virtual_thread_0/episode_30_reward_9841.04/',
            #'/data/svidchenko/afterlearning/moar_features_1/sgdr_1/saved_models/exploiting_virtual_thread_0/episode_66_reward_9851.11//',
            '/data/svidchenko/afterlearning/moar_features_1/sgdr_1/saved_models/exploiting_virtual_thread_0/episode_100_reward_9854.58/',
            '/data/svidchenko/afterlearning/moar_features_1/sgdr_1/saved_models/exploiting_virtual_thread_1/episode_73_reward_9867.67/',
            '/data/svidchenko/afterlearning/moar_features_1/sgdr_1/saved_models/exploiting_virtual_thread_0/episode_38_reward_9879.41//',
            '/data/svidchenko/afterlearning/moar_features_1/sgdr_1/saved_models/exploiting_virtual_thread_3/episode_9_reward_9881.80//'
        ]
    ]

    config = parse_config(directory)
    config['environment']['wrapper']['repeat_actions'] = 3

    submit(config, directories, repeats=1)
