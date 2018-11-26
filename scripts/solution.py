#!/usr/bin/env python
import traceback

# noinspection PyUnresolvedReferences
import gym_duckietown_agent  # DO NOT CHANGE THIS IMPORT (the environments are defined here)
from duckietown_challenges import wrap_solution, ChallengeSolution, ChallengeInterfaceSolution
from models.ddpg.model import load_model
from utils.env_wrappers import create_env
from utils.util import parse_config

MODEL_DIR = 'final_models/'
CONFIG_FILENAME = 'final_models/config.json'


def solve(params, cis):
    # python has dynamic typing, the line below can help IDEs with autocompletion
    assert isinstance(cis, ChallengeInterfaceSolution)
    # after this cis. will provide you with some autocompletion in some IDEs (e.g.: pycharm)
    cis.info('Creating model.')
    # you can have logging capabilties through the solution interface (cis).
    # the info you log can be retrieved from your submission files.
    model = load_model(MODEL_DIR, load_gpu_model_on_cpu=True)

    cis.info('Parsing config')
    config = parse_config(config_name=CONFIG_FILENAME)
    # We do not want to interrupt the submission env flow
    config["environment"]["wrapper"]["max_env_steps"] = 500001

    # We get environment from the Evaluation Engine
    cis.info('Making environment')
    internal_env_args = {'env_type': 'normal',
                         'env_init_args': {'name': params['env']},
                         'env_config': config['environment']['core']}
    env = create_env(config, internal_env_args, transfer=config['training']['transfer'])

    try:
        # Then we make sure we have a connection with the environment and it is ready to go
        cis.info('Reset environment')
        observation = env.reset()
        total_reward = 0

        # While there are no signal of completion (simulation done)
        # we run the predictions for a number of episodes, don't worry, we have the control on this part
        while True:
            # We tell the environment to perform this action and we get some info back in OpenAI Gym style
            action = model.act(observation)
            observation, (reward, _), done, info = env.step(action)

            # Here you may want to compute some stats, like how much reward are you getting
            # notice, this reward may no be associated with the challenge score.
            cis.info('action:')
            cis.info(action)
            total_reward += reward
            cis.info('total_reward')
            cis.info(total_reward)

            # It is important to check for this flag, the Evalution Engine will let us know when should we finish
            # if we are not careful with this the Evaluation Engine will kill our container and we will get no score
            # from this submission
            if 'simulation_done' in info:
                cis.info('simulation_done received.')
                break
            if done:
                cis.info('Episode done; calling reset()')
                observation = env.reset()
                total_reward = 0

    finally:
        # release CPU/GPU resources, let's be friendly with other users that may need them
        cis.info('Releasing resources')
    cis.info('Graceful exit of solve()')


class Submission(ChallengeSolution):
    def run(self, cis):
        assert isinstance(cis, ChallengeInterfaceSolution)  # this is a hack that would help with autocompletion

        # get the configuration parameters for this challenge
        params = cis.get_challenge_parameters()
        cis.info('Parameters: %s' % params)

        cis.info('Starting.')
        solve(params, cis)

        cis.set_solution_output_dict({})
        cis.info('Finished.')


if __name__ == '__main__':
    print('Starting submission')
    wrap_solution(Submission())
