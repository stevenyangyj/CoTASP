from copy import deepcopy

import jax
import numpy as np
import flax.linen as nn
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import Lasso


class BasisVectorLearner(object):

    def __init__(self, 
                 n_features: int, 
                 n_components: int, 
                 seed: int=0, 
                 scale: float=np.sqrt(2), 
                 verbose=False):
        self.rng = np.random.RandomState(seed)
        self._components = nn.initializers.orthogonal(scale)(
            jax.random.PRNGKey(seed), shape=(n_components, n_features))
        self._codes = np.zeros((1, n_components))
        self._n_components = n_components
        self._n_features = n_features
        self._n_samples = 0
        self._seed = seed
        self._verbose = verbose

        self.arch_samples = None

    def decompose(self, sample):
        assert sample.shape[1] == self._n_features
        self._n_samples += 1

        if self.arch_samples is None:
            self.arch_samples = sample
        else:
            self.arch_samples = np.vstack((self.arch_samples, sample))
        
        dict_learner = DictionaryLearning(
            max_iter=5000,
            alpha=0.1,
            n_components=self._n_components, 
            fit_algorithm='lars',
            code_init=self._codes,
            dict_init=self._components,
            transform_algorithm='lasso_lars', 
            transform_alpha=0.1,
            transform_max_iter=5000,
            random_state=self._seed
        )

        alphas = deepcopy(dict_learner.fit_transform(self.arch_samples))
        next_alphas = np.zeros((1, self._n_components))
        self._codes = np.vstack((alphas, next_alphas))
        self._components = deepcopy(dict_learner.components_)

        if self._verbose:
            recon = np.dot(alphas, self._components)
            print(f'Number of samples: {self._n_samples}')
            print(f'Level of sparsity: {np.mean(alphas == 0):.4f}')
            print(f'Recontruction loss: {np.mean((recon - self.arch_samples)**2)}')
            print('Samples:\n', self.arch_samples)
            print('Reconst:\n', recon)

    def get_components(self):
        return deepcopy(self._components)

    def get_next_codes(self):
        return deepcopy(self._codes[-1, :].reshape(1, -1))


class OnlineDictLearner(object):
    def __init__(self, 
                 n_features: int, 
                 n_components: int, 
                 seed: int=0,
                 scale: float=1.0,
                 verbose=False):

        self.N = 1
        # d = n_features, k = n_components
        self.D = np.eye(n_features)
        self.I = np.eye(n_features*n_components)
        self.A = np.zeros((n_features*n_components, n_features*n_components))
        self.b = np.zeros((n_features*n_components, 1))
        self.L = jax.nn.initializers.variance_scaling(scale, 'fan_in', 'normal')(
            jax.random.PRNGKey(seed), shape=(n_features, n_components))
        # self.L = (jax.random.uniform(jax.random.PRNGKey(seed), shape=(n_features, n_components)) - 0.5) * 2 * np.sqrt(1 / 12) * 1e-2
        self.s = None
        self.S = None
        self.arch_samples = None
        self._n_components = n_components
        self._verbose = verbose
        self.lasso_solver = Lasso(alpha=1e-5, fit_intercept=False, max_iter=5000, random_state=seed)

    def decompose(self, sample):
        self.lasso_solver.fit(self.L, sample.T)
        s = self.lasso_solver.coef_.reshape(-1,1)

        # collect coefs s:
        if self.S is None:
            self.S = s
            self.arch_samples = sample.T
        else:
            self.S = np.hstack([self.S, s])
            self.arch_samples = np.hstack([self.arch_samples, sample.T])

        # update stats
        self.A += np.kron(s.dot(s.T), self.D)
        self.b += np.kron(s.T, sample.dot(self.D)).T
        vals = np.linalg.inv(self.A / self.N + 1e-5 * self.I).dot(self.b / self.N)
        self.L = vals.reshape(self.L.shape, order='F')
        self.s = s

        if self._verbose:
            recon = np.dot(self.L, self.S)
            print(f'Number of samples: {self.N}')
            print(f'Level of sparsity: {np.mean(self.S == 0):.4f}')
            print(f'Recontruction loss: {np.mean((recon - self.arch_samples)**2)}')
            print('Samples:\n', self.arch_samples.T)
            print('Reconst:\n', recon.T)

        self.N += 1

    def get_components(self):
        return deepcopy(self.L.T)

    def get_codes(self):
        return deepcopy(self.s.T)

    def get_next_codes(self):
        return np.zeros((1, self._n_components))


if __name__ == "__main__":

    task_decomposer = OnlineDictLearner(
        n_features=5,
        n_components=5,
        seed=0,
        verbose=True
    )

    for i in range(10):
        # learnable parameters
        alphas = task_decomposer.get_next_codes()
        task_vectors = task_decomposer.get_components()

        # get imaginary sample
        new_sample = np.dot((np.random.uniform(size=alphas.shape) - 0.5) * 2 * 4, task_vectors)
        new_sample = (np.random.uniform(size=new_sample.shape) - 0.5) * 2 * 2

        # learning task vectors
        task_decomposer.decompose(new_sample)
    