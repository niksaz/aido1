import gym
import gym_duckietown_agent
import torch 
from duckietown_rl.env import launch_env
from duckietown_rl.ddpg import DDPG
from duckietown_rl.utils import evaluate_policy
from duckietown_rl.wrappers import NormalizeWrapper, ImgTransposer, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
import numpy as np


####### ========== IMPORTANT =========
# This will only run if you also start the container with
# the simulator and leave it running in the background like so
#
# docker run -tid -p 8902:8902 -p 5558:5558 -e DISPLAY=$DISPLAY -e DUCKIETOWN_CHALLENGE=LF --name gym-duckietown-server --rm -v /tmp/.X11-unix:/tmp/.X11-unix duckietown/gym-duckietown-server


experiment = 2
seed = 1
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
env = NormalizeWrapper(env)
env = ImgTransposer(env) # to make the images from 160x120x3 into 3x160x120
env = ActionWrapper(env)
# env = DtRewardWrapper(env) # not during testing

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
        while True:
            action = policy.predict(np.array(obs))
            obs, rew, done, misc = env.step(action)
            rewards.append(rew)
            env.render()
            if done:
                break
        print ("mean episode reward:",np.mean(rewards))