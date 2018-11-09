import math

import matplotlib.pyplot as plt
import numpy as np

from utils.util import parse_info


def visualize(log_names):
    i = 0
    x = list(range(1000))
    ys = []
    for log_name in log_names:
        log = parse_info(log_name)
        total_reward = 0.

        configs = [
            {'save_file': './videos/side_log_{}'.format(i), 'camera_rotation': [-0.3, 0., 0.]},
            {'save_file': './videos/front_log_{}'.format(i), 'camera_rotation': [-0.3, -math.pi / 2, 0.]},
            {'save_file': './videos/half_log_{}'.format(i), 'camera_rotation': [-0.3, -math.pi / 4, 0.]}
        ]
        i += 1
        #graphics = []
        #for config in configs:
        #    graphics.append(VirtualGraphics(config))

        y = [f['reward'] for f in log]
        ys.append(y)
        step = 0
        for log_frame in log:
            observation = log_frame['observation']
            reward = log_frame['reward']
            total_reward += reward
            #for gr in graphics:
            #    gr.refresh_frame(observation, reward, step)
            step += 1
            print(step, ":", reward, ":", 10. - np.linalg.norm(np.array(observation["target_vel"])[[0, 2]] -
                                                               np.array(observation["body_vel"]["pelvis"])[[0, 2]])**2)
        print(np.unique([l['observation']['target_vel'] for l in log], axis=0))
        print(total_reward)
    plt.figure()
    for y in ys:
        plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    log_names = ["submit_logs/first/log_0.json", "submit_logs/first/log_1.json"]
    visualize(log_names)
