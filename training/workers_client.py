import queue as py_queue
import time

import numpy as np

from models.ddpg.model import get_total_indices, find_index
from utils.buffers import create_buffer
from utils.util import set_seeds

MAGIC_SEED = 42497


def to_numpy(lst):
    return np.array(lst, np.float32)


def client_sampling_worker(config, p_id, global_update_step, sample_queues, episode_queue, filter_idx=None):
    set_seeds(p_id * MAGIC_SEED + (p_id if filter_idx is None else filter_idx))
    training_config = config['training']
    multimodel_buffering = config['model'].get("enable_multimodel", False) and \
                           not training_config.get("multimodel_parallelism_enabled", False)
    angle_dividers = config['model'].get('angle_dividers', None)
    speed_dividers = config['model'].get('speed_dividers', None)
    step_dividers = config['model'].get('step_dividers', None)
    buffer = [create_buffer(training_config) for _ in range(get_total_indices(angle_dividers, speed_dividers, step_dividers))] \
        if multimodel_buffering else create_buffer(training_config)
    received_examples = 1

    counter = 0

    while True:
        while True:
            try:
                replays = episode_queue.get_nowait()
                for (observation, action, reward, next_observation, done) in replays:
                    if multimodel_buffering:
                        buffer[find_index(observation[1], angle_dividers, speed_dividers, step_dividers)] \
                            .add(observation, action, reward, next_observation, done)
                    else:
                        if filter_idx is not None:
                            idx = find_index(observation[1], angle_dividers, speed_dividers, step_dividers)
                            observation = observation[0]
                            next_observation = next_observation[0]
                            if idx != filter_idx:
                                continue
                        buffer.add(observation, action, reward, next_observation, done)
                received_examples += len(replays)
            except py_queue.Empty:
                break

        if multimodel_buffering:
            sl = False
            for buff in buffer:
                sl = sl or len(buff) < training_config['batch_size']
            if sl:
                time.sleep(1)
                continue
        else:
            if len(buffer) < training_config['batch_size']:
                time.sleep(1)
                continue

        train_data_list = []

        for _ in range(len(sample_queues)):
            if multimodel_buffering:
                train_data = [buff.sample(batch_size=training_config['batch_size']) for buff in buffer]
            else:
                train_data = buffer.sample(batch_size=training_config['batch_size'])
            train_data_list.append(train_data)

        if counter % 10000 == 0:
            for sample_queue in sample_queues:
                print('sampling queue size: ', sample_queue.qsize())
            print()

        counter += 1

        buffer_size = len(buffer)

        for sample_queue, train_data in zip(sample_queues, train_data_list):
            sample_queue.put((train_data, received_examples, buffer_size))


def client_model_worker(model, observation_queue, action_queue):
    actions = []

    while True:
        observations = observation_queue.get()
        for (index, observation, noise) in observations:
            action = model.act(observation, noise)
            actions.append((index, action))
        action_queue.put(actions)
        actions = []


def client_observation_worker(observation_pipes, observation_queue):
    while True:
        observations = []

        for pipe_index in observation_pipes:
            if observation_pipes[pipe_index].poll():
                observation, noise = observation_pipes[pipe_index].recv()

                observations.append((pipe_index, observation, noise))

        if len(observations) == 0:
            continue

        observation_queue.put(observations)


def client_action_worker(action_pipes, action_queue):
    while True:
        actions = action_queue.get()

        for index, action in actions:
            action_pipes[index].send(action)
