from flask import Flask, request
# from . import utils
from utils.env_wrappers import ProstheticEnvironmentProxyWrapper
from utils.env_wrappers import ProstheticEnvironmentWrapper
from osim.env import ProstheticsEnv
import json
import gc
import argparse


def compress_observation(obs):
    obs_copy = dict(obs)
    del obs_copy['joint_pos']
    del obs_copy['joint_vel']
    del obs_copy['joint_acc']
    del obs_copy['body_vel']
    del obs_copy['body_acc']
    del obs_copy['body_acc_rot']
    del obs_copy['muscles']
    return obs_copy


worker = Flask(format('{module_name}').format(module_name=__name__))

env = ProstheticEnvironmentWrapper(visualize=False)


@worker.route('/post_step_request/', methods=['POST'])
def post_step_request():
    json_data = json.loads(request.get_json())
    observation, reward, done, info = env.step(**json_data)
    observation = compress_observation(observation)
    result = {'observation': observation, 'reward': reward, 'done': done, 'info': info}
    return json.dumps(result)


@worker.route('/post_reset_request/', methods=['POST'])
def post_reset_request():
    global env
    json_data = json.loads(request.get_json())
    observation = env.reset(**json_data)
    observation = compress_observation(observation)
    return json.dumps({'observation': observation})


@worker.route('/post_change_model_request/', methods=['POST'])
def post_change_model_request():
    json_data = json.loads(request.get_json())
    env.change_model(**json_data)
    return json.dumps({'success': True})


@worker.route('/post_collect_garbage_request/', methods=['POST'])
def post_collect_garbage_request():
    env.collect_garbage()
    return json.dumps({'success': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running server')
    parser.add_argument('--host', type=str, default='localhost', help='')
    parser.add_argument('--port', type=int, default=18000, help='')
    arguments = parser.parse_args()

    worker.run(host=arguments.host, port=arguments.port, debug=False)
