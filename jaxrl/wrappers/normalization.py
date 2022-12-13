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
    ):
        """This wrapper will normalize immediate rewards
        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
        """
        super().__init__(env)
        self.reward_rms = RunningMeanStd(shape=())
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, dones, infos = self.env.step(action)
        rews = np.array([rews])
        rews = self.normalize(rews)
        rews = rews[0]
        return obs, rews, dones, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.reward_rms.update(rews)
        return (rews - self.reward_rms.mean) / np.sqrt(self.reward_rms.var + self.epsilon)


if __name__ == "__main__":

    def print_reward(env: gym.Env):
        obs, done = env.reset(), False
        i = 0
        while not done:
            i += 1
            next_obs, rew, done, _ = env.step(env.action_space.sample())
            print(i, rew)

    env = gym.make('HalfCheetah-v3')
    env_wrapped = NormalizeReward(env)

    print_reward(env)
    print_reward(env_wrapped)
