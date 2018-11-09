from flask import Flask, request
from osim.env import ProstheticsEnv
import json
import gc


def compress_observation(observation):
    del observation['joint_pos']
    del observation['joint_vel']
    del observation['joint_acc']
    del observation['body_vel']
    del observation['body_acc']
    del observation['body_vel_rot']
    del observation['body_acc_rot']
    del observation['muscles']


def perform_command(command, command_out, result_in):
    json_data = json.loads(request.get_json())
    command_out.send((command, json_data))
    result = result_in.recv()
    return result


def server_web_worker(config, p_id, out_command, in_result):
    worker = Flask(format('{module_name}_{p_id}').format(module_name=__name__, p_id=p_id))

    server_config = config['training']['server']

    host_tcp = server_config['host_tcp']
    port_tcp = server_config['port_tcp_start'] + p_id

    @worker.route('/post_step_request/', methods=['POST'])
    def post_step_request():
        return perform_command('step', out_command, in_result)

    @worker.route('/post_reset_request/', methods=['POST'])
    def post_reset_request():
        return perform_command('reset', out_command, in_result)

    @worker.route('/post_change_model_request/', methods=['POST'])
    def post_change_model_request():
        return perform_command('change_model', out_command, in_result)

    @worker.route('/post_collect_garbage_request/', methods=['POST'])
    def post_collect_garbage_request():
        return perform_command('collect_garbage', out_command, in_result)

    worker.run(host=host_tcp, port=port_tcp, debug=False)


def server_env_worker(in_command, out_result):
    env = ProstheticsEnv(visualize=False)

    while True:
        command, args = in_command.recv()
        if command == 'step':
            observation, reward, done, info = env.step(**args)
            result = {'observation': observation, 'reward': reward, 'done': done, 'info': info}
            result = json.dumps(result)
        elif command == 'reset':
            observation = env.reset(**args)
            result = json.dumps({'observation': observation})
        elif command == 'change_model':
            env.change_model(**args)
            result = json.dumps({'success': True})
        # elif command == 'collect_garbage':
        #     del env
        #     gc.collect()
        #     env = ProstheticsEnv(visualize=False)
        #     result = json.dumps({'success': True})

        out_result.send(result)


def server_worker(config, p_id):
    worker = Flask(format('{module_name}_{p_id}').format(module_name=__name__, p_id=p_id))

    server_config = config['training']['server']

    host_tcp = server_config['host_tcp']
    port_tcp = server_config['port_tcp_start'] + p_id

    env = ProstheticsEnv(visualize=False)

    @worker.route('/post_step_request/', methods=['POST'])
    def post_step_request():
        json_data = json.loads(request.get_json())
        observation, reward, done, info = env.step(**json_data)
        result = {'observation': observation, 'reward': reward, 'done': done, 'info': info}
        return json.dumps(result)

    @worker.route('/post_reset_request/', methods=['POST'])
    def post_reset_request():
        json_data = json.loads(request.get_json())
        observation = env.reset(**json_data)
        return json.dumps({'observation': observation})

    @worker.route('/post_change_model_request/', methods=['POST'])
    def post_change_model_request():
        json_data = json.loads(request.get_json())
        env.change_model(**json_data)
        return json.dumps({'success': True})

    @worker.route('/post_collect_garbage_request/', methods=['POST'])
    def post_collect_garbage_request():
        nonlocal env
        del env
        gc.collect()
        env = ProstheticsEnv(visualize=False)
        return json.dumps({'success': True})

    worker.run(host=host_tcp, port=port_tcp, debug=False)
