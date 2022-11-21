from typing import Dict, List

import gym
import numpy as np
import jax.numpy as jnp


def evaluate(agent, env: gym.Env, num_episodes: int, with_task_embed=False, task_i=None) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    successes = None
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            if with_task_embed:
                action = agent.sample_a_with_task_embed(observation[np.newaxis], temperature=0.0)
            else:
                if task_i is None:
                    action = agent.sample_actions(observation[np.newaxis], temperature=0.0)
                else:
                    action = agent.sample_actions(observation[np.newaxis], task_i, temperature=0.0)
            observation, _, done, info = env.step(action)
        for k in stats.keys():
            stats[k].append(info['episode'][k])

        if 'success' in info:
            if successes is None:
                successes = 0.0
            successes += info['success']

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats

def evaluate_cl(agent, envs: List[gym.Env], num_episodes: int) -> Dict[str, float]:
    stats = {}
    sum_return = 0.0
    sum_success = 0.0
    list_log_keys = ['return']
    
    # dummy inputs
    # dummy_obs = jnp.ones((128, 12))

    for task_i, env in enumerate(envs):
        for k in list_log_keys:
            stats[f'{task_i}-{env.name}/{k}'] = []
        successes = None

        for _ in range(num_episodes):
            observation, done = env.reset(), False
            while not done:
                action = agent.sample_actions(observation[np.newaxis], task_i, temperature=0)
                action = np.asarray(action, dtype=np.float32).flatten()
                observation, _, done, info = env.step(action)
            for k in list_log_keys:
                stats[f'{task_i}-{env.name}/{k}'].append(info['episode'][k])

            if 'success' in info:
                if successes is None:
                    successes = 0.0
                successes += info['success']

        for k in list_log_keys:
            stats[f'{task_i}-{env.name}/{k}'] = np.mean(stats[f'{task_i}-{env.name}/{k}'])

        if successes is not None:
            stats[f'{task_i}-{env.name}/success'] = successes / num_episodes
            sum_success += stats[f'{task_i}-{env.name}/success']

        sum_return += stats[f'{task_i}-{env.name}/return']

        # stats[f'{task_i}-{env.name}/check_dummy_action'] = agent.sample_actions(dummy_obs, task_i, temperature=0).mean()

    stats['avg_return'] = sum_return / len(envs)
    stats['test/deterministic/average_success'] = sum_success / len(envs)

    return stats
