import numpy as np


class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing=int(1e5)):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2,
                 x0=None, size=1, sigma_min=None, n_steps_annealing=int(1e6)):
        super(OrnsteinUhlenbeckProcess, self).__init__(
            mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x.astype(np.float32)

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


def create_action_random_process(config):
    training_config = config['training']

    return OrnsteinUhlenbeckProcess(
        size=config['model']['num_action'],
        theta=training_config['rp_theta'],
        mu=training_config['rp_mu'],
        sigma=training_config['rp_sigma'],
        sigma_min=training_config['rp_sigma_min'])


def create_observation_random_process(config):
    training_config = config['training']

    return OrnsteinUhlenbeckProcess(
        size=config['model']['num_observations'],
        theta=training_config['rp_theta'],
        mu=training_config['rp_mu'],
        sigma=training_config['rp_sigma'],
        sigma_min=training_config['rp_sigma_min'])
