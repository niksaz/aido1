import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import scipy
import scipy.stats

from models.ddpg.model import load_model
from utils.env_wrappers import create_env
from utils.util import set_seeds, parse_config


def evaluate(config, directory):
    explorer_seed = config['training']['global_seed']
    set_seeds(explorer_seed)

    model = load_model(directory, load_gpu_model_on_cpu=True)

    internal_env_args = {'env_type': 'normal',
                         'env_init_args': {},
                         'env_config': config['environment']['core']}

    env = create_env(config, internal_env_args, transfer=config['training']['transfer'])

    reward_sum = 0.0
    reward_modified_sum = 0.0

    done = True
    j = 0

    while True:
        if done or j > 100:
            observation = env.reset()
            env.env.env.render()
            done = False
            j = 0
            print('reset environment')

        action = model.act(observation)
        print('action', action)

        observation, (reward, reward_modified), done, _ = env.step(action)
        env.env.env.render()

        reward_sum += reward
        reward_modified_sum += reward_modified

        print('{} {:.2f} {:.2f} {:.2f}'.format(j, reward, reward_modified, reward_sum))
        j += config["environment"]["wrapper"]["repeat_actions"]


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


if __name__ == '__main__':
    directory = 'final_models/'

    config = parse_config(directory=directory)
    config["environment"]["wrapper"]["max_env_steps"] = 1000
    evaluate(config, directory)
