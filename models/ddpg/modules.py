from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


def get_module(config):
    if config['name'] == 'linear':
        return LinearWrapper(**config['args'])
    if config['name'] == 'flatten':
        return Flatten()
    if config['name'] == 'conv_2d':
        return Conv2dWrapper(**config['args'])
    if config['name'] == 'max_pooling_2d':
        return nn.MaxPool2d(**config['args'])
    if config['name'] == 'batch_norm_2d':
        return nn.BatchNorm2d(**config['args'])
    if config['name'] == 'layer_norm':
        return nn.LayerNorm(**config['args'])
    if config['name'] == 'dropout':
        return nn.Dropout(**config['args'])
    if config['name'] == 'leaky_relu':
        return nn.LeakyReLU()
    if config['name'] == 'elu':
        return nn.ELU()
    if config['name'] == 'tanh':
        return nn.Tanh()
    raise AssertionError('Unknown module named ' + config['name'])


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class LinearWrapper(nn.Module):
    def __init__(self, in_features, out_features, weight_init):
        super(LinearWrapper, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.weight_init = weight_init
        self.init_weights()

    def init_weights(self):
        if self.weight_init == "fanin":
            self.linear.weight.data = fanin_init(self.linear.weight.data.size())
        if self.weight_init == "uniform":
            init_w = 3e-3
            self.linear.weight.data.uniform_(-init_w, init_w)
        if self.weight_init == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.linear.weight.data)
        if self.weight_init == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.linear.weight.data)
        if self.weight_init == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.linear.weight.data)
        if self.weight_init == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(self.linear.weight.data)

    def forward(self, input):
        x = self.linear.forward(input)
        # print('LINEAR', x)
        return x


class Conv2dWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, weight_init):
        super(Conv2dWrapper, self).__init__()
        self.kernel = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        self.weight_init = weight_init
        self.init_weights()

    def init_weights(self):
        if self.weight_init == "fanin":
            self.kernel.weight.data = fanin_init(self.kernel.weight.data.size())
        if self.weight_init == "uniform":
            init_w = 3e-3
            self.kernel.weight.data.uniform_(-init_w, init_w)
        if self.weight_init == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.kernel.weight.data)
        if self.weight_init == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.kernel.weight.data)

    def forward(self, input):
        return self.kernel.forward(input)


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        self.input_nets = nn.ModuleList()
        self.output_nets = nn.ModuleList()

        self._build(config)

    def forward(self, *inputs):
        input_forwards = [input_net(input) for input_net, input in zip(self.input_nets, inputs)]
        intermediate_result = torch.cat(input_forwards, dim=1)
        results = [output_net(intermediate_result) for output_net in self.output_nets]
        return results

    def _build(self, config):
        for module_config in config:
            if module_config['name'] in ('inputs', 'outputs'):
                for module_config_rec in module_config['modules']:
                    net = self._build_simple_net(module_config_rec)
                    if module_config['name'] == 'inputs':
                        self.input_nets.append(net)
                    if module_config['name'] == 'outputs':
                        self.output_nets.append(net)

    @staticmethod
    def _build_simple_net(config):
        net = MetaNet()

        last_shape = None

        for module_config in config:
            module_name = module_config['name']
            if module_name == 'input':
                last_shape = module_config['in_features']
                continue
            if module_name == 'input_channeled':
                last_shape = module_config['in_channels']
                continue

            if module_name == 'linear':
                module_config['args']['in_features'] = last_shape
                last_shape = module_config['args']['out_features']
            elif module_name == 'conv_2d':
                module_config['args']['in_channels'] = last_shape
                last_shape = module_config['args']['out_channels']
            elif module_name == 'batch_norm_2d':
                if 'args' not in module_config.keys():
                    module_config['args'] = {}
                    module_config['args']['num_features'] = last_shape
            elif module_config["name"] == 'layer_norm':
                if 'args' not in module_config.keys():
                    module_config['args'] = {}
                    module_config['args']['normalized_shape'] = last_shape
            elif module_name == 'flatten':
                last_shape = module_config['args']['out_features']

            module = get_module(module_config)

            net.append_module(module)

        return net


class MetaNet(nn.Module):
    def __init__(self):
        super(MetaNet, self).__init__()
        self.internal_modules = nn.ModuleList()

    def append_module(self, module):
        self.internal_modules.append(module)

    def forward(self, x):
        for module in self.internal_modules:
            x = module.forward(x)
        return x

    def __iter__(self):
        for module in self.internal_modules:
            yield module


class Actor(nn.Module):
    def __init__(self, actor_config):
        super(Actor, self).__init__()
        self.net = Net(actor_config)

    def forward(self, *inputs):
        x = self.net.forward(*inputs)
        return x[0]

    def set_inputs(self, source_actor):
        self.net.input_nets = deepcopy(source_actor.net.input_nets)

    def set_outputs(self, source_actor):
        self.net.output_nets = deepcopy(source_actor.net.output_nets)


class Critic(nn.Module):
    def __init__(self, critic_config):
        super(Critic, self).__init__()
        self.net = Net(critic_config)

    def forward(self, *inputs):
        x = self.net.forward(*inputs)
        return x[0]

    def set_inputs(self, source_critic):
        self.net.input_nets = deepcopy(source_critic.net.input_nets)

    def set_outputs(self, source_critic):
        self.net.output_nets = deepcopy(source_critic.net.output_nets)
