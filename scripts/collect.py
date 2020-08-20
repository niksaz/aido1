import argparse
import os
import shutil

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import cv2

from models.ddpg.model import load_model
from utils.env import launch_env
from utils.util import set_seeds, parse_config
from utils.reward_shaping.env_utils import Transformer, PreliminaryTransformer


def ask_to_remove_if_exists(directory):
    if os.path.exists(directory):
      should_remove = input(f'The directory {directory} exists. Should it be removed? (y/n): ')
      if should_remove.lower() == 'y':
        shutil.rmtree(directory)


def collect(model_directory, output_dir, episodes, max_steps):
    config = parse_config(directory=model_directory)
    global_seed = config['training']['global_seed']
    set_seeds(global_seed)
    model = load_model(model_directory, load_gpu_model_on_cpu=True)

    ask_to_remove_if_exists(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    env = launch_env()
    env_seed = config['environment']['core'].get('seed', global_seed)
    env.seed(env_seed)
    preprocessor = PreliminaryTransformer()
    transformer = Transformer()

    total_samples = 0
    for episode in range(episodes):
        print('resetting environment')
        raw_observation = env.reset()
        preprocessed_observation = preprocessor.transform(raw_observation)
        transformer.reset(preprocessed_observation)
        transformed_observation = transformer.transform(preprocessed_observation)
        reward_sum = 0.0
        for step in range(max_steps):
            action = model.act(transformed_observation)
            print('action', action)

            sample_filename = f'ep_{episode:02}_{step:03}'
            img_path = os.path.join(output_dir, sample_filename + '.png')
            cv2.imwrite(img_path, cv2.cvtColor(raw_observation, cv2.COLOR_RGB2BGR))
            action_path = os.path.join(output_dir, sample_filename + '.npy')
            np.save(action_path, action.astype(np.float32))
            total_samples += 1

            raw_observation, reward, done, info = env.step(action)
            preprocessed_observation = preprocessor.transform(raw_observation)
            transformed_observation = transformer.transform(preprocessed_observation)

            reward_sum += reward

            print(f'step={step} rew={reward:.2f} rew_sum={reward_sum:.2f}')
            if done:
                break
    print(f'Collected {total_samples} samples in total. The expected number was {episodes * max_steps}.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_directory', type=str, default='final_models')
    parser.add_argument('--output_directory', type=str, default='samples')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=256)
    args = parser.parse_args()

    collect(args.model_directory, args.output_directory, args.episodes, args.max_steps)


if __name__ == '__main__':
    main()
