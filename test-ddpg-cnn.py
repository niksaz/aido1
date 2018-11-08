import numpy as np
import torch

from duckietown_rl.args import get_ddpg_args_test
from duckietown_rl.config import REPEAT
from duckietown_rl.ddpg import DDPG
from duckietown_rl.env import launch_env
from duckietown_rl.wrappers import ImgTransposer, ImgStacker, \
    GrayscaleWrapper, ActionWrapper, ResizeWrapper, SteeringToWheelVelWrapper

args = get_ddpg_args_test()

experiment = args.experiment
seed = args.seed
policy_name = "DDPG"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


file_name = "{}_{}_{}".format(
    policy_name,
    experiment,
    seed
)

# Launch the env with our helper function
env = launch_env()

# Wrappers
env = ResizeWrapper(env)
env = GrayscaleWrapper(env)
env = ImgTransposer(env)
env = ImgStacker(env)

env = ActionWrapper(env)
env = SteeringToWheelVelWrapper(env)


state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize policy
policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")

policy.load(file_name, directory="./pytorch_models")

with torch.no_grad():
    while True:
        obs = env.reset()
        env.render()
        rewards = []
        done = False
        for i in range(args.max_timesteps):
            if done:
                break
            action = policy.predict(np.array(obs))
            for _ in range(REPEAT):
                print('ACTION', i, action)
                obs, rew, done, misc = env.step(action)
                rewards.append(rew)
                env.render()
                if done:
                    break
        print("mean episode reward:", np.mean(rewards))
