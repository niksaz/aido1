# Author: Mikita Sazanovich

import matplotlib.pyplot as plt
import os
import cv2

from features.straight.straight import line_approx
from utils.reward_shaping.env_utils import PreliminaryTransformer


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


if __name__ == '__main__':
  main()
