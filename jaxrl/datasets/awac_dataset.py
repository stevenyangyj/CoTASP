import os

import d4rl
import gym
import numpy as np

from jaxrl.datasets.dataset import Batch, Dataset

# awac_demos corresponds to expert demonstrations.
# awac_off corresponds to additional data collected
# with BC trained on demonstrations.
ENV_NAME_TO_FILE = {
    'HalfCheetah-v2': {
        'awac_off': 'hc_off_policy_15_demos_100.npy',
        'awac_demo': 'hc_action_noise_15.npy'
    },
    'Walker2d-v2': {
        'awac_off': 'walker_off_policy_15_demos_100.npy',
        'awac_demo': 'walker_action_noise_15.npy'
    },
    'Ant-v2': {
        'awac_off': 'ant_off_policy_15_demos_100.npy',
        'awac_demo': 'ant_action_noise_15.npy'
    }
}


class AWACDataset(Dataset):

    def __init__(self,
                 env_name: str,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):

        # Reuse d4rl path for now.
        dataset_path = os.path.join(d4rl.offline_env.DATASET_PATH, 'avac')
        zip_path = os.path.join(dataset_path, 'all.zip')

        url = 'https://drive.google.com/u/0/uc?id=1edcuicVv2d-PqH1aZUVbO5CeRq3lqK89'
        gdown.cached_download(url, zip_path, postprocess=gdown.extractall)

        observations = []
        actions = []
        rewards = []
        terminals = []
        dones_float = []
        next_observations = []

        env = gym.make(env_name)
        # Contacentate both datasets for now.
        for dataset_name in ['awac_off', 'awac_demo']:
            file_name = ENV_NAME_TO_FILE[env_name][dataset_name]

            dataset = np.load(os.path.join(dataset_path, file_name),
                              allow_pickle=True)

            for trajectory in dataset:
                if len(trajectory['observations']) == env._max_episode_steps:
                    trajectory['terminals'][-1] = False

                observations.append(trajectory['observations'])
                actions.append(trajectory['actions'])
                rewards.append(trajectory['rewards'])
                terminals.append(trajectory['terminals'])
                done_float = np.zeros_like(trajectory['rewards'])
                done_float[-1] = 1.0
                dones_float.append(done_float)
                next_observations.append(trajectory['next_observations'])

        observations = np.concatenate(observations, 0)
        actions = np.concatenate(actions, 0)
        rewards = np.concatenate(rewards, 0)
        terminals = np.concatenate(terminals, 0)
        dones_float = np.concatenate(dones_float, 0)
        next_observations = np.concatenate(next_observations, 0)

        if clip_to_eps:
            lim = 1 - eps
            actions = np.clip(actions, -lim, lim)

        super().__init__(observations=observations.astype(np.float32),
                         actions=actions.astype(np.float32),
                         rewards=rewards.astype(np.float32),
                         masks=1.0 - terminals.astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=next_observations.astype(
                             np.float32),
                         size=len(observations))
