import gzip
import json
import os
import pickle
import random

import numpy as np
import torch


def make_dir_if_required(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_linear_decay_fn(initial_value, final_value, max_step):
    def decay_fn(step):
        relative = 1. - step / max_step
        return initial_value * relative + final_value * (1. - relative)

    return decay_fn


def create_cycle_decay_fn(initial_value, final_value, cycle_len, num_cycles):
    max_step = cycle_len * num_cycles

    def decay_fn(step):
        relative = 1. - step / max_step
        relative_cosine = 0.5 * (np.cos(np.pi * np.mod(step, cycle_len) / cycle_len) + 1.0)
        return relative_cosine * (initial_value - final_value) * relative + final_value

    return decay_fn


def create_exponential_decay_fn(initial_value, final_value, max_step, updates):
    update_coeff = (final_value / initial_value) ** (1 / updates)
    update_steps = max_step / updates

    def decay_fn(step):
        return initial_value * (update_coeff ** int(step/update_steps))

    return decay_fn


def create_cosine_cyclic_decay_fn(initial_value, final_value, period_base, period_modifier=1):
    period_modifier_log = np.log(period_modifier)

    def decay_fn(step):
        if abs(period_modifier_log) > 1e-3 and step // period_base > 0:
            cycle_len = period_base * period_modifier ** (int(np.log(step / period_base) / period_modifier_log))
        else:
            cycle_len = period_base
        relative_cosine = 0.5 * (1 + np.cos(np.pi * (step % cycle_len) / cycle_len))
        return final_value + (initial_value - final_value) * relative_cosine
    return decay_fn


def create_decay_fn(decay_type, **kwargs):
    if decay_type == "linear":
        return create_linear_decay_fn(**kwargs)
    elif decay_type == "cycle":
        return create_cycle_decay_fn(**kwargs)
    elif decay_type == "exponential":
        return create_exponential_decay_fn(**kwargs)
    elif decay_type == "cyclic_cosine":
        return create_cosine_cyclic_decay_fn(**kwargs)
    else:
        raise NotImplementedError()


class TrainingDecay:
    def __init__(self, config):
        self.decays = {}
        for name in config:
            self.decays[name] = create_decay_fn(config[name]['type'], **config[name]['args'])
        self.realization = {}

    def __call__(self, params):
        for name in self.realization:
            params[name] = self.realization[name]

    def update_step(self, step):
        for name in self.decays:
            self.realization[name] = self.decays[name](step)


def cut_off_leg(action):
    cut_off_leg_action = np.zeros(19, np.float32)
    cut_off_leg_action[:8] = action[:8]
    cut_off_leg_action[8:] = action[11:]
    return cut_off_leg_action


def serialize(data):
    serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    serialized = gzip.compress(serialized)
    serialized = str(serialized, 'utf-8')
    return serialized


def parse_config(directory=".", config_name='config.json'):
    with open(directory + "/" + config_name) as f:
        json_config = json.load(f)
    return json_config


def parse_info(filename):
    with open(filename) as f:
        json_config = json.load(f)
    return json_config


def save_info(filename, info):
    with open(filename, 'w') as f:
        json.dump(info, f)


def from_numpy(data):
    if isinstance(data, np.ndarray):
        return [from_numpy(x) for x in data]
    if isinstance(data, np.inexact):
        return float(data)
    if isinstance(data, np.integer):
        return int(data)
    return data


def deserialize(data):
    deserialized = bytes(data, 'utf-8')
    deserialized = gzip.decompress(deserialized)
    deserialized = pickle.loads(deserialized)
    return deserialized
