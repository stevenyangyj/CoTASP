from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Set of wrappers for normalizing actions and observations."""
import numpy as np

import gym


# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
    ):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeReward(gym.core.Wrapper):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.
    The exponential moving average will have variance :math:`(1 - \gamma)^2`.
    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 1e-8,
        reward_alpha=0.001,
    ):
        """This wrapper will normalize immediate rewards
        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
        """
        super().__init__(env)
        self.epsilon = epsilon
        self._reward_alpha = reward_alpha
        self._reward_mean = 0.
        self._reward_var = 1.

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, dones, infos = self.env.step(action)
        rews = self._apply_normalize_reward(rews)
        return obs, rews, dones, infos

    def _update_reward_estimate(self, reward):
        self._reward_mean = (1 - self._reward_alpha) * \
            self._reward_mean + self._reward_alpha * reward
        self._reward_var = (
            1 - self._reward_alpha
        ) * self._reward_var + self._reward_alpha * np.square(
            reward - self._reward_mean)

    def _apply_normalize_reward(self, reward):
        """Compute normalized reward.
        Args:
            reward (float): Reward.
        Returns:
            float: Normalized reward.
        """
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + self.epsilon)
    
    
class RescaleReward(gym.core.Wrapper):
    '''
    This wrapper will rescale immediate rewards based on a constant factor.
    '''
    def __init__(self, env: gym.Env, reward_scale: float = 1.0):
        super().__init__(env)
        self.reward_scale = reward_scale
        
    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        rews = rews * self.reward_scale
        return obs, rews, dones, infos


if __name__ == "__main__":

    def print_reward(env: gym.Env):
        obs, done = env.reset(), False
        i = 0
        while not done:
            i += 1
            next_obs, rew, done, _ = env.step(env.action_space.sample())
            print(i, rew)

    env = gym.make('Hopper-v3')
    env_wrapped = NormalizeReward(env)

    print_reward(env)
    print_reward(env_wrapped)
