import abc
import copy
import os
import subprocess
import sys
from multiprocessing import set_start_method

import torch
import torch.multiprocessing as torch_mp

try:
    set_start_method('spawn')
except RuntimeError:
    pass

from multiprocessing import Value
from models.ddpg.model import create_model, RemoteModel, load_model
from training.explorers import explore_single_thread
from training.trainers import train_single_thread

from training.workers_client import (client_sampling_worker,
                                     client_model_worker,
                                     client_observation_worker,
                                     client_action_worker)


class BaseManager(metaclass=abc.ABCMeta):
    def __init__(self):
        self.processes = []

    @abc.abstractmethod
    def manage(self):
        pass

    def _get_in_connections(self, connection_tuples):
        return self._get_connections(connection_tuples, 0)

    def _get_out_connections(self, connection_tuples):
        return self._get_connections(connection_tuples, 1)

    @staticmethod
    def _get_connections(connection_tuples, index):
        connections = {i: connection_tuples[i][index] for i in range(len(connection_tuples))}
        return connections

    def _join_processes(self):
        for p in self.processes:
            p.join()

    def _terminate_processes(self):
        for p in self.processes:
            p.terminate()


class TrainManager(BaseManager):
    def __init__(self, config):
        super(TrainManager, self).__init__()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.config = config
        self.training_config = self.config['training']

        if self.training_config['transfer']:
            self.target_model = load_model(self.training_config['model_path'])
        else:
            self.target_model = create_model(self.config['model'])
            alt_model = None
            if self.training_config.get('init_actor', False):
                alt_model = load_model(self.training_config['model_path'])
                self.target_model.init_actor(alt_model)
            if self.training_config.get('init_critic', False):
                if alt_model is None:
                    alt_model = load_model(self.training_config['model_path'])
                self.target_model.init_critic(alt_model)
            self._prime_model(self.target_model, device)

        self.models = []
        self.proxy_models = []

        for _ in range(self.training_config['num_threads_training']):
            model = copy.deepcopy(self.target_model)
            proxy_model = copy.deepcopy(self.target_model)
            self._prime_model(model, device)
            self._prime_model(proxy_model, device)
            self.models.append(model)
            self.proxy_models.append(proxy_model)

        self.processes = []

        self.episode_queues = [torch_mp.Queue() for _ in range(self.training_config['num_threads_sampling'])]

        self.sample_queues = [torch_mp.Queue(maxsize=self.training_config['sampling_queue_max_len']) for _ in
                  range(self.training_config['num_threads_training'])]

        self.action_conns = [torch_mp.Pipe(duplex=False) for _ in
                             range(self.training_config['num_threads_exploring_virtual'])]

        self.observation_conns = [torch_mp.Pipe(duplex=False) for _ in
                                  range(self.training_config['num_threads_exploring_virtual'])]

        self.observation_queue = torch_mp.Queue()
        self.action_queue = torch_mp.Queue()

        self.start_barrier = torch_mp.Barrier(self.training_config['num_threads_training'])
        self.finish_barrier = torch_mp.Barrier(self.training_config['num_threads_training'])
        self.update_lock = torch_mp.Lock()

        self.best_reward = Value('f', 0.0)
        self.global_episode = Value('i', 0)
        self.global_update_step = Value('i', 0)

    def manage(self):
        try:
            self._explore()
            self._exploit()
            self._sample()
            self._train()
            self._run_virtual_models()
            self._explore_virtual()
            self._exploit_virtual()
            self._join_processes()
        except KeyboardInterrupt:
            self._terminate_processes()

    def _explore(self):
        episode_queues = self.episode_queues

        for p_id in range(self.training_config['num_threads_exploring']):
            internal_env_args = {'env_type': 'normal',
                                 'env_init_args': {},
                                 'env_config': self.config['environment']['core']}

            p = torch_mp.Process(
                target=explore_single_thread,
                args=('exploration', self.config, p_id,
                      self.models[p_id % self.training_config['num_threads_training']],
                      internal_env_args, episode_queues, self.best_reward,
                      self.global_episode, self.global_update_step)
            )
            p.start()
            self.processes.append(p)

    def _exploit(self):
        episode_queues = self.episode_queues

        for p_id in range(self.training_config['num_threads_exploiting']):
            internal_env_args = {'env_type': 'normal',
                                 'env_init_args': {},
                                 'env_config': self.config['environment']['core']}

            p = torch_mp.Process(
                target=explore_single_thread,
                args=('exploiting', self.config, p_id, self.target_model, internal_env_args,
                      episode_queues, self.best_reward,
                      self.global_episode, self.global_update_step)
            )
            p.start()
            self.processes.append(p)

    def _sample(self):
        for p_id, sample_queue, episode_queue in zip(range(self.training_config['num_threads_sampling']),
                                                     self.sample_queues, self.episode_queues):
            p = torch_mp.Process(
                target=client_sampling_worker,
                args=(self.config, p_id, self.global_update_step, [sample_queue], episode_queue)
            )
            p.start()
            self.processes.append(p)

    def _train(self):
        for p_id, sample_queue in zip(range(self.training_config['num_threads_training']), self.sample_queues):
            p = torch_mp.Process(
                target=train_single_thread,
                args=(self.config, p_id, self.target_model, self.proxy_models[p_id], self.models,
                       self.start_barrier, self.finish_barrier, self.update_lock,
                      sample_queue, self.best_reward, self.global_episode, self.global_update_step)
            )
            p.start()
            self.processes.append(p)

    def _run_virtual_models(self):
        for p_id in range(self.training_config['num_threads_model_workers']):
            p = torch_mp.Process(
                target=client_model_worker,
                args=(self.models[p_id % self.training_config['num_threads_training']],
                      self.observation_queue, self.action_queue)
            )
            p.start()
            self.processes.append(p)

        in_observation_conns = self._get_in_connections(self.observation_conns)

        p = torch_mp.Process(
            target=client_observation_worker,
            args=(in_observation_conns, self.observation_queue)
        )
        p.start()
        self.processes.append(p)

        out_action_conns = self._get_out_connections(self.action_conns)

        p = torch_mp.Process(
            target=client_action_worker,
            args=(out_action_conns, self.action_queue)
        )
        p.start()
        self.processes.append(p)

    def _explore_virtual(self):
        out_observation_conns = self._get_out_connections(self.observation_conns)
        in_action_conns = self._get_in_connections(self.action_conns)

        episode_queues = self.episode_queues

        for p_id in range(self.training_config['num_threads_exploring_virtual']):
            model = RemoteModel(in_action_conn=in_action_conns[p_id],
                                out_observation_conn=out_observation_conns[p_id])

            internal_env_args = {'env_type': 'virtual',
                                 'env_init_args': {
                                     'host_tcp': self.training_config['client']['host_tcp'],
                                     'port_tcp': self.training_config['client']['port_tcp_start'] + p_id
                                 },
                                 'env_config': self.config['environment']['core']
                                 }

            p_id += self.training_config['num_threads_exploring']

            p = torch_mp.Process(
                target=explore_single_thread,
                args=('exploration_virtual', self.config, p_id, model, internal_env_args,
                      episode_queues, self.best_reward,
                      self.global_episode, self.global_update_step)
            )
            p.start()
            self.processes.append(p)

    def _exploit_virtual(self):

        episode_queues = self.episode_queues

        for p_id in range(self.training_config['num_threads_exploiting_virtual']):
            internal_env_args = {'env_type': 'virtual',
                                 'env_init_args': {
                                     'host_tcp': self.training_config['client']['host_tcp'],
                                     'port_tcp': self.training_config['client']['port_tcp_start'] +
                                                 self.training_config['num_threads_exploring_virtual'] + p_id
                                 },
                                 'env_config': self.config['environment']['core']
                                 }

            p_id += self.training_config['num_threads_exploiting']

            p = torch_mp.Process(
                target=explore_single_thread,
                args=('exploiting_virtual', self.config, p_id, self.target_model, internal_env_args,
                      episode_queues, self.best_reward,
                      self.global_episode, self.global_update_step)
            )
            p.start()
            self.processes.append(p)

    @staticmethod
    def _prime_model(model, device):
        model.train()
        model.share_memory()
        model.to(device)


class VirtualManager(BaseManager):
    def __init__(self, config):
        super(VirtualManager, self).__init__()

        self.config = config
        self.training_config = self.config['training']

        self.processes = []

    def manage(self):
        try:
            self._run_server_workers()
            self._join_processes()
        except KeyboardInterrupt:
            self._terminate_processes()

    def _join_processes(self):
        for p in self.processes:
            p.wait()

    def _terminate_processes(self):
        for p in self.processes:
            p.kill()

    def _run_server_workers(self):
        host = self.training_config['server']['host_tcp']
        port_start = self.training_config['server']['port_tcp_start']
        env_accuracy = self.config['environment']['env_accuracy']

        sys.path.append(os.getcwd())

        for p_id in range(self.training_config['num_threads_exploring_virtual'] +
                                  self.training_config['num_threads_exploiting_virtual']):
            process = subprocess.Popen(['python', 'pyramid_worker.py',
                                        '--host', host,
                                        '--port', str(port_start + p_id),
                                        '--accuracy', str(env_accuracy)])
            self.processes.append(process)
