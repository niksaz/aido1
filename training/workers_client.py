import queue as py_queue
import time

import numpy as np

from utils.buffers import create_buffer
from utils.util import set_seeds

MAGIC_SEED = 42497


def to_numpy(lst):
    return np.array(lst, np.float32)


def client_sampling_worker(config, p_id, global_update_step, sample_queues, episode_queue, filter_idx=None):
    set_seeds(p_id * MAGIC_SEED + (p_id if filter_idx is None else filter_idx))
    training_config = config['training']
    buffer = create_buffer(training_config)
    received_examples = 1

    counter = 0

    while True:
        taken_replays = 0
        while True:
            if taken_replays > 128:
                print("Episode queue is too big!")
                episode_queue.clear()
                break
            try:
                replays = episode_queue.get_nowait()
                for (observation, action, reward, next_observation, done) in replays:
                    buffer.add(observation, action, reward, next_observation, done)
                received_examples += len(replays)
                taken_replays += 1
            except py_queue.Empty:
                break

        if len(buffer) < training_config['batch_size']:
            time.sleep(1)
            continue

        train_data_list = []

        for _ in range(len(sample_queues)):
            train_data = buffer.sample(batch_size=training_config['batch_size'])
            train_data_list.append(train_data)

        counter += 1

        buffer_size = len(buffer)

        for sample_queue, train_data in zip(sample_queues, train_data_list):
            sample_queue.put((train_data, received_examples, buffer_size))
        print('Put samples')


def client_model_worker(model, observation_queue, action_queue):
    actions = []

    while True:
        indexes = []
        observations = observation_queue.get()
        for (index, observation, noise) in observations:
            action = model.act(observation, noise)
            indexes.append(index)
            actions.append((index, action))
        print("Model acts for {}".format(indexes))
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

        print("Deliver observations to model worker")
        observation_queue.put(observations)


def client_action_worker(action_pipes, action_queue):
    while True:
        actions = action_queue.get()

        print("Sending actions to peers")
        for index, action in actions:
            action_pipes[index].send(action)

