import random
from multiprocessing import set_start_method

import numpy as np
import scipy
import scipy.stats
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue

from models.ddpg.model import load_model, AverageModel, RemoteModel, create_model, average_update
from training.workers_client import client_model_worker, client_observation_worker, client_action_worker
from utils.env_wrappers import create_env
from utils.util import set_seeds, parse_config, save_info

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def _get_connections(connection_tuples, index):
    connections = {i: connection_tuples[i][index] for i in range(len(connection_tuples))}
    return connections


def _get_in_connections(connection_tuples):
    return _get_connections(connection_tuples, 0)


def _get_out_connections(connection_tuples):
    return _get_connections(connection_tuples, 1)


def evaluate_single_thread(p_id, model, config, seeds_per_thread, output: Queue):
    rewards = []
    modified_rewards = []
    steps_counts = []
    infos = []
    for seed_plus in range(p_id * seeds_per_thread, (p_id + 1) * seeds_per_thread):
        explorer_seed = 721 + seed_plus * 29
        set_seeds(explorer_seed)

        internal_env_args = {'env_type': 'virtual',
                             'env_init_args': {
                                 'host_tcp': config['training']['client']['host_tcp'],
                                 'port_tcp': config['training']['client']['port_tcp_start'] + p_id
                             },
                             'env_config': config['environment']['core']
                             }
        internal_env_args['env_config']['seed'] = explorer_seed

        env = create_env(config, internal_env_args, transfer=config['training']['transfer'])
        observation = env.reset()

        done = False
        steps = 0
        reward_sum = 0.0
        reward_modified_sum = 0.0

        while not done:
            observation_transformed, _ = observation

            observation, (reward, reward_modified), done, _ = env.step(model.act(observation_transformed))

            reward_sum += reward
            reward_modified_sum += reward_modified

            steps += config["environment"]["wrapper"]["repeat_actions"]
        target_velocities = [[float(v) for v in tv]
                             for tv in np.unique([obs["target_vel"]
                                                  for obs in env.observations], axis=0)]
        velocity_similarity_measure = [np.linalg.norm(np.array(obs["target_vel"])[[0, 2]]
                                                      - np.array(obs["body_vel"]["pelvis"])[[0, 2]])
                                       for obs in env.observations]
        velocity_confidence_intervals = [mean_confidence_interval(velocity_similarity_measure, 0.95),
                                         mean_confidence_interval(velocity_similarity_measure, 0.99)]
        rewards.append(reward_sum)
        modified_rewards.append(reward_modified_sum)
        steps_counts.append(steps)
        print(explorer_seed, ':', reward_sum, ':', steps)
        infos.append({"target": target_velocities,
                      "target_similarity_confidence_intervals": velocity_confidence_intervals,
                      "seed": explorer_seed})
    output.put((rewards, modified_rewards, steps_counts, infos))


def evaluate(config, directories, seeds_per_thread=5, repeats=1, model_workers=None, average_weights=False):
    models = []

    if model_workers is None:
        model_workers = config['training']['num_threads_model_workers']
    threads = config['training']['num_threads_exploring_virtual'] + config['training']['num_threads_exploiting_virtual']

    for model_directory in directories:
        models.append(load_model(model_directory))

    if average_weights:
        average_model = create_model(config['model'])
        average_model.train()
        average_model.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        #print("start averaging")
        average_update(average_model.actor, [model.actor for model in models])
        average_model = AverageModel([average_model], config, repeats)
    else:
        average_model = AverageModel(models, config, repeats)

    processes = []
    results = Queue()
    observation_queue = Queue()
    action_queue = Queue()
    observation_conns = [mp.Pipe(duplex=False) for _ in
                         range(threads)]
    action_conns = [mp.Pipe(duplex=False) for _ in
                    range(threads)]
    try:
        for p_id in range(model_workers):
            p = mp.Process(
                target=client_model_worker,
                args=(average_model,
                      observation_queue, action_queue)
            )
            p.start()
            processes.append(p)

        in_observation_conns = _get_in_connections(observation_conns)
        out_observation_conns = _get_out_connections(observation_conns)

        p = mp.Process(
            target=client_observation_worker,
            args=(in_observation_conns, observation_queue)
        )
        p.start()
        processes.append(p)

        in_action_conns = _get_in_connections(action_conns)
        out_action_conns = _get_out_connections(action_conns)

        p = mp.Process(
            target=client_action_worker,
            args=(out_action_conns, action_queue)
        )
        p.start()
        processes.append(p)

        for p_id in range(threads):
            p = mp.Process(
                target=evaluate_single_thread,
                args=(
                    p_id,
                    RemoteModel(in_action_conns[p_id], out_observation_conns[p_id]),
                    config,
                    seeds_per_thread,
                    results
                )
            )
            p.start()
            processes.append(p)

        rewards_total = []
        rewards_without_falling = []
        modified_rewards_total = []
        step_counts_total = []
        infos_total = []
        for _ in range(threads):
            rewards, modified_rewards, step_counts, infos = results.get()
            rewards_total += rewards
            modified_rewards_total += modified_rewards
            step_counts_total += step_counts
            infos_total += infos
            for r, s in zip(rewards, step_counts):
                if s > 999:
                    rewards_without_falling.append(r)
    finally:
        for p in processes:
            p.terminate()
    return rewards_total, modified_rewards_total, step_counts_total, infos_total, rewards_without_falling


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def all_combinations(model_sets, start=2):
    result = []
    for n in range(start, len(model_sets) + 1):
        result += _all_combinations(model_sets, n)
    return result


def _all_combinations(model_sets, n):
    if n == 0:
        return [[]]
    results = [[r + [model_sets[i]] for r in _all_combinations(model_sets[i + 1:], n - 1)] for i in
               range(len(model_sets) - n + 1)]
    result = []
    for r in results:
        result += r
    return result


def print_model(info):
    print("Models:")
    for i in info['models']:
        print(" ", i)
    print("Repeats:", info["repeats"])
    print("Rewards:")
    print("  Mean confidence interval (0.95):", info["rewards"][0])
    print("  Mean confidence interval (0.99):", info["rewards"][1])
    print("  Quantile (0.05, 0.1, 0.25, 0.5):", info["rewards"][2])
    print("Rewards without falling:")
    print("  Mean confidence interval (0.95):", info["rewards_without_falling"][0])
    print("  Mean confidence interval (0.99):", info["rewards_without_falling"][1])
    print("  Quantile (0.05, 0.1, 0.25, 0.5):", info["rewards_without_falling"][2])
    print("Falling rate:")
    print("  Below 300 steps:", info["falling_rate"]["300"])
    print("  Below 600 steps:", info["falling_rate"]["600"])
    print("  Below 900 steps:", info["falling_rate"]["900"])
    print("  Below 1000 steps:", info["falling_rate"]["1000"])
    print()
    print()


def multitest(model_sets, log, averaging_models=(2, 3, 4), seeds_per_thread=4, repeat_set=(1, 2), model_workers=10,
              average_weights=False):
    total_results = []
    model_sets = all_combinations(model_sets)
    # model_sets = list(reversed([[s] for s in model_sets]))
    random.shuffle(model_sets)
    random.shuffle(model_sets)
    random.shuffle(model_sets)
    try:
        for directories in model_sets:
            for repeats in repeat_set:
                directory = directories[0]
                dirs = list(reversed(directories))
                config = parse_config(directory)

                rewards, modified_rewards, step_counts, infos, rws = evaluate(config, dirs,
                                                                              seeds_per_thread=seeds_per_thread,
                                                                              repeats=repeats,
                                                                              model_workers=model_workers,
                                                                              average_weights=average_weights)
                logging_list = []

                for r, mr, sc, inf in zip(rewards, modified_rewards, step_counts, infos):
                    logging_list.append({
                        "reward": r,
                        "modified_reward": mr,
                        "step_count": sc,
                        "info": inf
                    })

                total_results.append({
                    "models": dirs,
                    "repeats": repeats,
                    "result": logging_list,
                    "rewards": [mean_confidence_interval(rewards, confidence=0.95),
                                mean_confidence_interval(rewards, confidence=0.99),
                                np.quantile(rewards, [0.05, 0.1, 0.25, 0.5])],
                    "rewards_without_falling": [mean_confidence_interval(rws, confidence=0.95),
                                                mean_confidence_interval(rws, confidence=0.99),
                                                np.quantile(rws, [0.05, 0.1, 0.25, 0.5])],
                    "falling_rate": {
                        "300": len([st for st in step_counts if st < 300]) / len(step_counts),
                        "600": len([st for st in step_counts if st < 600]) / len(step_counts),
                        "900": len([st for st in step_counts if st < 900]) / len(step_counts),
                        "1000": len([st for st in step_counts if st < 1000]) / len(step_counts)
                    }
                })

                for i in range(10):
                    print()
                print_model(total_results[-1])
                print('='*20)
                print()
                print()

                for res in sorted(total_results, key=lambda res: res["rewards_without_falling"][0][0], reverse=True)[
                           :min(len(total_results), 5)]:
                    print_model(res)

                for i in range(10):
                    print()
    except Exception as e:
        print("Exception", e)
        pass
    finally:
        for i in range(10):
            print()
        print("Results:")
        for res in total_results:
            print_model(res)

        save_info(log, total_results)


if __name__ == '__main__':
    directories = [
        '/data/svidchenko/middle_learning/more_features_3/all9/saved_models/exploiting_virtual_thread_1/episode_15_reward_9909.03',
        '/data/svidchenko/middle_learning/more_features_3/all9/saved_models/exploiting_virtual_thread_1/episode_25_reward_9883.13',
        '/data/svidchenko/middle_learning/more_features_3/all9/saved_models/exploiting_virtual_thread_1/episode_55_reward_9917.39',
        '/data/svidchenko/middle_learning/more_features_3/all9/saved_models/exploiting_virtual_thread_1/episode_60_reward_9922.19',
        '/data/svidchenko/middle_learning/more_features_3/all9/saved_models/exploiting_virtual_thread_1/episode_65_reward_9929.47',
        '/data/svidchenko/middle_learning/more_features_3/all9/saved_models/exploiting_virtual_thread_1/episode_75_reward_9914.62',
        '/data/svidchenko/middle_learning/more_features_3/all9/saved_models/exploiting_virtual_thread_1/episode_80_reward_9897.23',
        #'/data/svidchenko/middle_learning/more_features_3/all8/saved_models/exploiting_virtual_thread_0/episode_40_reward_9933.77'
    ]
    multitest(directories, "/data/svidchenko/testing_logs/final.json", seeds_per_thread=1, repeat_set=(1,), average_weights=False)
