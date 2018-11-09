import os

import numpy as np
import torch
from hyperdash import Experiment

from duckietown_rl import utils
from duckietown_rl.args import get_ddpg_args_train
from duckietown_rl.config import REPEAT
from duckietown_rl.ddpg import DDPG
from duckietown_rl.env import launch_env
from duckietown_rl.utils import seed, evaluate_policy
from duckietown_rl.wrappers import ImgTransposer, ImgStacker, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper, GrayscaleWrapper, \
    SteeringToWheelVelWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment = 3
policy_name = "DDPG"
exp = Experiment("[duckietown] - ddpg")


args = get_ddpg_args_train()

file_name = "{}_{}_{}".format(
    policy_name,
    experiment,
    str(args.seed),
)

if not os.path.exists("./results"):
    os.makedirs("./results")

models_dir = "./pytorch_models"
if args.save_models and not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Launch the env with our helper function
env = launch_env()

# Wrappers
env = ResizeWrapper(env)
env = GrayscaleWrapper(env)
env = ImgTransposer(env)
env = ImgStacker(env)

env = ActionWrapper(env)

env = SteeringToWheelVelWrapper(env)

# Set seeds
seed(args.seed)

state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])


# Initialize policy
policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
try:
    policy.load(filename=file_name, directory=models_dir)
except FileNotFoundError as err:
    print('No previous model weights exist:', err.filename)
    print('Starting from random init')

replay_buffer = utils.ReplayBuffer(args.replay_buffer_max_size)

# Evaluate untrained policy
evaluations = []

evaluations.append(evaluate_policy(env, policy))
exp.metric("rewards", evaluations[0])

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
episode_reward = None
env_counter = 0
while total_timesteps < args.max_timesteps:

    if done:
        if total_timesteps != 0:
            print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))
            policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

        # Evaluate episode
        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval %= args.eval_freq
            evaluations.append(evaluate_policy(env, policy))
            exp.metric("rewards", evaluations[-1])
            np.savez("./results/{}.npz".format(file_name),evaluations)

        if args.save_models:
            policy.save(file_name, directory="./pytorch_models")

        # Reset environment
        env_counter += 1
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # Select action randomly or according to policy
    if total_timesteps < args.start_timesteps:
        action = env.action_space.sample()
    else:
        action = policy.predict(np.array(obs))
        if args.expl_noise != 0:
            action = (action + np.random.normal(
                0,
                args.expl_noise,
                size=env.action_space.shape[0])
                      ).clip(env.action_space.low, env.action_space.high)

    old_obs = obs
    repeat_reward = 0
    for _ in range(REPEAT):
        if done:
            break
        # Perform selected action
        obs, reward, done, _ = env.step(action)
        repeat_reward += reward

    print('action reward', action, repeat_reward)
    episode_reward += repeat_reward
    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1

    if episode_timesteps >= args.env_timesteps:
        done = True

    # Store data in replay buffer
    done_float = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
    replay_buffer.add(old_obs, obs, action, repeat_reward, done_float)


# Final evaluation
evaluations.append(evaluate_policy(env, policy))
exp.metric("rewards", evaluations[-1])

if args.save_models:
    policy.save(file_name, directory="./pytorch_models")
np.savez("./results/{}.npz".format(file_name),evaluations)

exp.end()
