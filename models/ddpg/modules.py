import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class NoisyLinear(nn.Module):
    """Applies a noisy linear transformation to the incoming data:
    :math:`y = (mu_w + sigma_w \cdot epsilon_w)x + mu_b + sigma_b \cdot epsilon_b`
    More details can be found in the paper `Noisy Networks for Exploration` _ .
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: True
        factorised: whether or not to use factorised noise. Default: True
        std_init: initialization constant for standard deviation component of weights. If None,
            defaults to 0.017 for independent and 0.4 for factorised. Default: None
    Shape:
        - Input: :math:`(N, in\_features)`
        - Output: :math:`(N, out\_features)`
    Attributes:
        weight: the learnable weights of the module of shape (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
    Examples::
        >>> m = nn.NoisyLinear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True, factorised=True, std_init=None,
                 enable_random=True, cut_random=False):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.factorised = factorised
        self.enable_random = enable_random
        self.cut_random = cut_random
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = Parameter(torch.Tensor(out_features, in_features))
        # self.weight_normal = torch.distributions.normal.Normal(self.weight_mu, self.weight_sigma)

        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_sigma = Parameter(torch.Tensor(out_features))
            # self.bias_normal = torch.distributions.normal.Normal(self.bias_mu, self.bias_sigma)
        else:
            self.register_parameter('bias', None)

        if not std_init:
            if self.factorised:
                self.std_init = 0.4
            else:
                self.std_init = 0.017
        else:
            self.std_init = std_init

        self.reset_parameters(bias)

    def reset_parameters(self, bias):
        if self.factorised:
            mu_range = 1. / math.sqrt(self.weight_mu.size(1))

            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

            if bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
        else:
            mu_range = math.sqrt(3. / self.weight_mu.size(1))

            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init)

            if bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init)

    def scale_noise(self, size):
        x = torch.Tensor(size).normal_().cuda() if torch.cuda.is_available() else torch.Tensor(size).normal_().cpu()
        x = x.sign().mul(x.abs().sqrt())
        return x

    def forward(self, input):
        if not self.enable_random:
            weight_epsilon = torch.Tensor(self.out_features, self.in_features).fill_(0.).cuda() \
                if torch.cuda.is_available() else torch.Tensor(self.out_features, self.in_features).fill_(0.).cpu()
            bias_epsilon = torch.Tensor(self.out_features).fill_(0.).cuda() \
                if torch.cuda.is_available() else torch.Tensor(self.out_features).fill_(0.).cpu()
        elif self.factorised:
            epsilon_in = self.scale_noise(self.in_features)
            epsilon_out = self.scale_noise(self.out_features)
            weight_epsilon = epsilon_out.ger(epsilon_in)
            bias_epsilon = self.scale_noise(self.out_features)
        else:
            weight_epsilon = torch.Tensor(self.out_features, self.in_features).normal_().cuda() \
                if torch.cuda.is_available() else torch.Tensor(self.out_features, self.in_features).normal_().cpu()
            bias_epsilon = torch.Tensor(self.out_features).normal_().cuda() \
                if torch.cuda.is_available() else torch.Tensor(self.out_features).normal_().cpu()
            # weight_epsilon = torch.Tensor(self.out_features, self.in_features).normal_()
            # bias_epsilon = torch.Tensor(self.out_features).normal_()
        if self.cut_random:
            weight_epsilon.clamp_(-1, 1)
            bias_epsilon.clamp_(-1, 1)

        return F.linear(input,
                        self.weight_mu + self.weight_sigma.mul(weight_epsilon),
                        self.bias_mu + self.bias_sigma.mul(bias_epsilon))

        # weight_normal = torch.distributions.normal.Normal(self.weight_mu, self.weight_sigma)
        # bias_normal = torch.distributions.normal.Normal(self.bias_mu, self.bias_sigma)
        #
        # return F.linear(input, weight_normal.sample(), bias_normal.sample())

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


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
        return self.linear.forward(input)


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
        x = x.view(x.size()[0], -1)
        return x


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_module(config):
    if config['name'] == 'noise_linear':
        return NoisyLinear(**config['args'])
    if config['name'] == 'linear':
        return LinearWrapper(**config['args'])
    if config['name'] == 'layer_norm':
        return nn.LayerNorm(**config['args'])
    if config['name'] == 'conv2d':
        return Conv2dWrapper(**config['args'])
    if config['name'] == 'maxpool2d':
        return nn.MaxPool2d(**config['args'])
    if config['name'] == 'flatten':
        return Flatten()
    if config['name'] == 'identity':
        return Identity()
    if config['name'] == 'elu':
        return nn.ELU()
    if config['name'] == 'relu':
        return nn.ReLU()
    if config['name'] == 'sigmoid':
        return nn.Sigmoid()
    if config['name'] == 'tanh':
        return nn.Tanh()


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

    def _build_simple_net(self, config):
        net = MetaNet()

        last_shape = None

        for module_config in config:
            if module_config['name'] == 'input':
                last_shape = module_config['in_features']
                continue

            if module_config['name'] in ('linear', 'noise_linear'):
                module_config['args']['in_features'] = last_shape
                last_shape = module_config['args']['out_features']

            elif module_config['name'] in ('conv2d',):
                module_config['args']["in_channels"] = last_shape
                last_shape = module_config['args']['out_channels']

            elif module_config["name"] == 'layer_norm':
                if 'args' not in module_config.keys():
                    module_config['args'] = {}
                    module_config['args']['normalized_shape'] = last_shape

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

    def forward(self, observation, action):
        obs_act = torch.cat((observation, action), dim=1)
        x = self.net.forward(obs_act)
        return x[0]

    def set_inputs(self, source_critic):
        self.net.input_nets = deepcopy(source_critic.net.input_nets)

    def set_outputs(self, source_critic):
        self.net.output_nets = deepcopy(source_critic.net.output_nets)


class DecoyActor(nn.Module):
    def __init__(self):
        super(DecoyActor, self).__init__()

        self.internal_modules = nn.ModuleList()

        self.internal_modules.append(NoisyLinear(in_features=480, out_features=512, bias=True, factorised=False))
        self.internal_modules.append(nn.ELU())
        self.internal_modules.append(nn.LayerNorm(normalized_shape=512))

        self.internal_modules.append(NoisyLinear(in_features=512, out_features=512, bias=True, factorised=False))
        self.internal_modules.append(nn.ELU())
        self.internal_modules.append(nn.LayerNorm(normalized_shape=512))

        self.internal_modules.append(LinearWrapper(in_features=512, out_features=512, weight_init='xavier_normal'))
        self.internal_modules.append(nn.ELU())
        self.internal_modules.append(nn.LayerNorm(normalized_shape=512))

        self.internal_modules.append(LinearWrapper(in_features=512, out_features=512, weight_init='xavier_normal'))
        self.internal_modules.append(nn.ELU())
        self.internal_modules.append(nn.LayerNorm(normalized_shape=512))

        self.internal_modules.append(LinearWrapper(in_features=512, out_features=19, weight_init='xavier_uniform'))
        self.internal_modules.append(nn.Sigmoid())

    def forward(self, x):
        return self._forward(x)

    def _forward(self, x):
        for module in self.internal_modules:
            x = module.forward(x)
        return x


class DecoyCritic(nn.Module):
    def __init__(self):
        super(DecoyCritic, self).__init__()

        self.internal_modules = nn.ModuleList()

        self.internal_modules.append(LinearWrapper(in_features=499, out_features=1024, weight_init='xavier_normal'))
        self.internal_modules.append(nn.ELU())
        self.internal_modules.append(nn.LayerNorm(normalized_shape=1024))

        self.internal_modules.append(LinearWrapper(in_features=1024, out_features=1024, weight_init='xavier_normal'))
        self.internal_modules.append(nn.ELU())
        self.internal_modules.append(nn.LayerNorm(normalized_shape=1024))

        self.internal_modules.append(LinearWrapper(in_features=1024, out_features=1024, weight_init='xavier_normal'))
        self.internal_modules.append(nn.ELU())
        self.internal_modules.append(nn.LayerNorm(normalized_shape=1024))

        self.internal_modules.append(LinearWrapper(in_features=1024, out_features=1024, weight_init='xavier_normal'))
        self.internal_modules.append(nn.ELU())
        self.internal_modules.append(nn.LayerNorm(normalized_shape=1024))

        self.internal_modules.append(LinearWrapper(in_features=1024, out_features=1, weight_init='xavier_uniform'))

    def forward(self, observation, action):
        x = torch.cat((observation, action), dim=1)
        x = self._forward(x)
        return x

    def _forward(self, x):
        for module in self.internal_modules:
            x = module.forward(x)
        return x
