import json
import argparse
from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.view import view_config
from pyramid.response import Response


# def compress_observation(obs):
#     obs_copy = dict(obs)
#     del obs_copy['joint_pos']
#     del obs_copy['joint_vel']
#     del obs_copy['joint_acc']
#     del obs_copy['body_vel']
#     del obs_copy['body_acc']
#     del obs_copy['body_acc_rot']
#     del obs_copy['muscles']
#     return obs_copy
from utils.env_wrappers import DuckietownEnvironmentWrapper

env = None


def post_step_request(request):
    json_data = json.loads(request.json_body)
    observation, reward, done, info = env.step(**json_data)
    # observation = compress_observation(observation)
    result = {'observation': observation, 'reward': reward, 'done': done, 'info': info}
    return Response(json=result)


def post_reset_request(request):
    print(request.json_body)
    json_data = json.loads(request.json_body)
    observation = env.reset(**json_data)
    # observation = compress_observation(observation)
    result = {'observation': observation}
    return Response(json=result)


def post_change_model_request(request):
    json_data = json.loads(request.json_body)
    env.change_model(**json_data)
    result = {'success': True}
    return Response(json=result)


def post_collect_garbage_request(request):
    env.collect_garbage()
    result = json.dumps({'success': True})
    return Response(json=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running server')
    parser.add_argument('--host', type=str, default='localhost', help='')
    parser.add_argument('--port', type=int, default=18000, help='')
    arguments = parser.parse_args()

    env = DuckietownEnvironmentWrapper()

    with Configurator() as config:
        config.add_route('post_step_request', '/post_step_request/')
        config.add_view(post_step_request, route_name='post_step_request', request_method='POST')

        config.add_route('post_reset_request', '/post_reset_request/')
        config.add_view(post_reset_request, route_name='post_reset_request', request_method='POST')

        config.add_route('post_change_model_request', '/post_change_model_request/')
        config.add_view(post_change_model_request, route_name='post_change_model_request', request_method='POST')

        config.add_route('post_collect_garbage_request', '/post_collect_garbage_request/')
        config.add_view(post_collect_garbage_request, route_name='post_collect_garbage_request', request_method='POST')

        app = config.make_wsgi_app()

    server = make_server(arguments.host, arguments.port, app)
    server.serve_forever()
