import gym
import numpy as np


def launch_env(id=None, should_patch_opposite_direction_lane=False):
    if id is None:
        from gym_duckietown.simulator import Simulator
        env = Simulator(
            seed=123, # random seed
            map_name="loop_empty",
            max_steps=500001, # we don't want the gym to reset itself
            domain_rand=0,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4, # start close to straight
            full_transparency=True,
            distortion=True,
        )
    else:
        env = gym.make(id)

    if should_patch_opposite_direction_lane:
        from types import MethodType
        env.closest_curve_point = MethodType(closest_curve_by_distance_point, env)

    return env

from gym_duckietown.graphics import bezier_closest, bezier_point, bezier_tangent
def closest_curve_by_distance_point(self, pos, angle=None):
    i, j = self.get_grid_coords(pos)
    curves = self._get_tile(i, j)['curves']
    distances = [np.linalg.norm(bezier_point(curve, bezier_closest(curve, pos)) - pos) for curve in curves]
    current_curve = np.argmin(distances)
    cps = curves[current_curve]

    # Find closest point and tangent to this curve
    t = bezier_closest(cps, pos)
    point = bezier_point(cps, t)
    tangent = bezier_tangent(cps, t)

    return point, tangent

def patched_env_test():
    import os
    import cv2
    import random

    RANDOM_STATE = 42
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    image_dir = 'tests/patched_env/'
    os.makedirs(image_dir, exist_ok=True)

    original_env = launch_env()
    original_env.seed(RANDOM_STATE)
    patched_env = launch_env(should_patch_opposite_direction_lane=True)
    patched_env.seed(RANDOM_STATE)

    for i in range(5):
        original_observation = original_env.reset()
        patched_observation = patched_env.reset()

        if not np.allclose(original_observation, patched_observation):
            print(i)
            cv2.imwrite(f'{image_dir}{i}_original.png', cv2.cvtColor(original_observation, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'{image_dir}{i}_patched.png', cv2.cvtColor(patched_observation, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    patched_env_test()
