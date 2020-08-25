# Author: Mikita Sazanovich

import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

from features.straight.straight import line_approx
from utils.reward_shaping.env_utils import PreliminaryTransformer
from utils.env import launch_env


def main():
  preliminary_transformer = PreliminaryTransformer()

  image_dir = 'samples'
  image_files = sorted(os.listdir(image_dir))
  for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame)
    plt.waitforbuttonpress()
    # transformed_frame = line_approx(frame)
    transformed_frame = preliminary_transformer.transform(frame)
    plt.imshow(transformed_frame[0, :, :], cmap='gray', vmin=0, vmax=255)
    plt.waitforbuttonpress()

def save_observation_as_image(obs, filename, image_dir='tests/colorspace/', colorspace_conversion=None):
    if obs.shape[0] == 1:
        # to save in grayscale we need to remove channel
        obs = np.squeeze(obs, axis=0)
    elif obs.shape[0] == 3:
        # to save in rgb we need to transpose observation (channel last)
        obs = np.transpose(obs, axes=(1, 2, 0))
    if colorspace_conversion:
        obs = cv2.cvtColor(obs, colorspace_conversion)

    os.makedirs(image_dir, exist_ok=True)
    cv2.imwrite(f'{image_dir}{filename}', obs)

def colorspace_test():
    env = launch_env()

    observation = env.reset()

    save_observation_as_image(observation, filename='original_observation.png', colorspace_conversion=cv2.COLOR_RGB2BGR)

    print(f'original shape: {observation.shape}')

    segmentation_transformer = PreliminaryTransformer(use_segmentation=True)
    segmentation_obs = segmentation_transformer.transform(observation)
    save_observation_as_image(segmentation_obs, filename='with_segmentation_observation.png')

    grayscale_transformer = PreliminaryTransformer(use_segmentation=False, colorspace_conversion=cv2.COLOR_RGB2GRAY)
    grayscale_obs = grayscale_transformer.transform(observation)
    save_observation_as_image(grayscale_obs, filename='grayscale_observation.png')

    rgb_transformer = PreliminaryTransformer(use_segmentation=False, colorspace_conversion=None)
    rgb_obs = rgb_transformer.transform(observation)
    save_observation_as_image(rgb_obs, filename='rgb_observation.png', colorspace_conversion=cv2.COLOR_RGB2BGR)

    hsv_transformer = PreliminaryTransformer(use_segmentation=False, colorspace_conversion=cv2.COLOR_RGB2HSV)
    hsv_obs = hsv_transformer.transform(observation)
    save_observation_as_image(hsv_obs, filename='hsv_observation.png', colorspace_conversion=cv2.COLOR_HSV2BGR)

    yuv_transformer = PreliminaryTransformer(use_segmentation=False, colorspace_conversion=cv2.COLOR_RGB2YUV)
    yuv_obs = yuv_transformer.transform(observation)
    save_observation_as_image(yuv_obs, filename='yuv_observation.png', colorspace_conversion=cv2.COLOR_YUV2BGR)


def create_from_config_test():
    segmentation_transformer = PreliminaryTransformer(use_segmentation=True, colorspace_conversion=None)
    grayscale_transformer = PreliminaryTransformer(use_segmentation=False, colorspace_conversion='COLOR_RGB2GRAY')
    rgb_transformer = PreliminaryTransformer(use_segmentation=False, colorspace_conversion=None)
    hsv_transformer = PreliminaryTransformer(use_segmentation=False, colorspace_conversion='COLOR_RGB2HSV')
    yuv_transformer = PreliminaryTransformer(use_segmentation=False, colorspace_conversion='COLOR_RGB2YUV')


if __name__ == '__main__':
    # main()
    colorspace_test()
    create_from_config_test()
