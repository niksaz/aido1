import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import scipy
import scipy.stats
import math

from models.ddpg.model import load_model, ChooseRandomModel
from utils.env_wrappers import create_env
from utils.util import set_seeds, parse_config


def evaluate(config, directory, directories, seed_plus):
    explorer_seed = config['training']['global_seed'] + seed_plus * 29
    set_seeds(explorer_seed)

    directories.append(directory)
    models = []

    for model_directory in directories:
        models.append(load_model(model_directory))

    model = ChooseRandomModel(models)

    # env = create_env(config['environment'], visualize=True, adapting=True)
    # config['environment']['wrapper']['features']['body_rot_relative'] = ["pelvis", "torso", "head"]
    # config['environment']['wrapper']['repeat_actions'] = 3

    internal_env_args = {'env_type': 'normal',
                         'env_init_args': {
                             'env_type': 'normal',
                             'env_init_args': {
                                 'visualize': False,
                                 'integrator_accuracy': 5e-4
                             },
                             'visualizers_configs': [
                                 {'save_file': './videos/side_{}'.format(seed_plus), 'camera_rotation': [-0.3, 0., 0.]},
                                 {'save_file': './videos/front_{}'.format(seed_plus),
                                  'camera_rotation': [-0.3, -math.pi / 2, 0.]},
                                 {'save_file': './videos/half_{}'.format(seed_plus),
                                  'camera_rotation': [-0.3, -math.pi / 4, 0.]}
                             ]
                         },
                         'env_config': {
                             "model": "3D",
                             "prosthetic": True,
                             "difficulty": 1,
                             "max_steps": 1000,
                             'seed': explorer_seed}
                         }

    env = create_env(config, internal_env_args, transfer=config['training']['transfer'])

    # config['environment']['core']['prosthetic'] = True
    # config['environment']['wrapper']['repeat_frames'] = 1
    # env = create_env(config['environment'], visualize=True, transfer=True)

    reward_sum = 0.0
    reward_modified_sum = 0.0

    observation = env.reset()

    replays_list = []

    repeats = 1
    done = False
    j = 0

    while not done:
        observation_transformed, _ = observation

        action = model.act(observation_transformed)

        observation, (reward, reward_modified), done, _ = env.step(action)

        reward_sum += reward
        reward_modified_sum += reward_modified

        # print(j, reward, reward_modified, reward_sum, reward_modified_sum)
        print('{} {:.2f} {:.2f}'.format(j, reward, reward_modified, reward_sum))
        j += config["environment"]["wrapper"]["repeat_actions"]
        # if j == 2:
        #     break
        if done:
            print(np.unique(np.array(list(map(lambda obs: obs["target_vel"], env.observations))), axis=0))
            return reward_sum


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


if __name__ == '__main__':

    # 2403, 2411, 2394, 2331
    # directory = '/home/ivan/data/prosthetics/exps/6/saved_models/exploiting_common_thread_0/episode_647_reward_2409.39/'
    #
    # directories = [
    #     # '/home/ivan/data/prosthetics/exps/6/saved_models/exploiting_common_thread_0/episode_676_reward_2406.04/',
    #     # '/home/ivan/data/prosthetics/exps/6/saved_models/exploiting_common_thread_0/episode_684_reward_2407.29/',
    #     # '/home/ivan/data/prosthetics/exps/6/saved_models/exploiting_common_thread_0/episode_696_reward_2414.30/'
    # ]

    directory = './../../logdir/saved_models/moar_features_1/'
    # directory = './saved_models/exploiting_thread_1/episode_516_reward_9465.81/'

    directories = [
        # './../../logdir/saved_models/coors_rotation/second',
        # './../../logdir/saved_models/coors_rotation/third'
    ]

    config = parse_config()
    config["environment"]["wrapper"]["target_transformer_config"]["noise"] = 0.
    config["environment"]["wrapper"]["target_transformer_type"] = "normal"
    config["environment"]["wrapper"]["repeat_actions"] = 3
    config["environment"]["core"]["max_steps"] = 1000
    config["environment"]["wrapper"]["reward_aggregations"] = [{
        "class": "TransformAndBound",
        "config": {
            "move": -19.0,
            "scale": 2.0,
            "bound": 1.0
        }
    }]
    rewards = []

    for seed_plus in range(40):
        print('seed_plus: ', seed_plus)
        rewards.append(evaluate(config, directory, directories, seed_plus))
        print('avg reward: {:.2f}'.format(sum(rewards) / len(rewards)))
        print()

    # rewards = [8583.006289442048, 4185.446762271158, 4316.1490028889475, 5662.302282981055, 6106.8078176703075,
    #            4103.217939933044, 4135.65101632019, 3830.298518980987, 7685.510861573788, 3788.757992492399]

    print("Repeat actions: {}".format(config["environment"]["wrapper"]["repeat_actions"]))
    print(rewards)
    print(mean_confidence_interval(rewards, confidence=0.95))
    print(mean_confidence_interval(rewards, confidence=0.99))
