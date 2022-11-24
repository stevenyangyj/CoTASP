from copy import deepcopy

import jax
import numpy as np
import flax.linen as nn
from scipy import spatial
from sklearn.decomposition import DictionaryLearning, sparse_encode
from sklearn.linear_model import Lasso
from sklearn.utils import check_array, check_random_state

from plot_utils import heatmap, annotate_heatmap


def _update_dict(
    dictionary,
    Y,
    code,
    A=None,
    B=None,
    verbose=False,
    random_state=None,
    positive=False,
):
    """Update the dense dictionary factor in place.

    Parameters
    ----------
    dictionary : ndarray of shape (n_components, n_features)
        Value of the dictionary at the previous iteration.

    Y : ndarray of shape (n_samples, n_features)
        Data matrix.

    code : ndarray of shape (n_samples, n_components)
        Sparse coding of the data against which to optimize the dictionary.

    A : ndarray of shape (n_components, n_components), default=None
        Together with `B`, sufficient stats of the online model to update the
        dictionary.

    B : ndarray of shape (n_features, n_components), default=None
        Together with `A`, sufficient stats of the online model to update the
        dictionary.

    verbose: bool, default=False
        Degree of output the procedure will print.

    random_state : int, RandomState instance or None, default=None
        Used for randomly initializing the dictionary. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    positive : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20
    """
    n_samples, n_components = code.shape
    random_state = check_random_state(random_state)

    if A is None:
        A = code.T @ code
    if B is None:
        B = Y.T @ code

    n_unused = 0

    for k in range(n_components):
        if A[k, k] > -np.inf:
            # 1e-6 is arbitrary but consistent with the spams implementation
            # -np.inf means that never resample atoms.
            dictionary[k] += (B[:, k] - A[k] @ dictionary) / A[k, k]
        else:
            # kth atom is almost never used -> sample a new one from the data
            newd = Y[random_state.choice(n_samples)]

            # add small noise to avoid making the sparse coding ill conditioned
            noise_level = 0.01 * (newd.std() or 1)  # avoid 0 std
            noise = random_state.normal(0, noise_level, size=len(newd))

            dictionary[k] = newd + noise
            code[:, k] = 0
            n_unused += 1

        if positive:
            np.clip(dictionary[k], 0, None, out=dictionary[k])

        # Projection on the constraint set ||V_k|| <= 1
        dictionary[k] /= max(np.linalg.norm(dictionary[k]), 1)

    if verbose and n_unused > 0:
        print(f"{n_unused} unused atoms resampled.")

    return dictionary


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

    
class OnlineDictLearnerV2(object):
    def __init__(self, 
                 n_features: int, 
                 n_components: int,
                 seed: int=0,
                 scale: float=1.0,
                 alpha: float=1e-3,
                 method: str='lasso_lars', # ['lasso_cd', 'lasso_lars', 'threshold']
                 positive_code: bool=False,
                 verbose=False):

        self.N = 0
        self.rng = np.random.RandomState(seed=seed)
        self.A = np.zeros((n_components, n_components))
        self.B = np.zeros((n_features, n_components))
        dictionary = self.rng.normal(loc=0.0, scale=scale, size=(n_components, n_features))
        dictionary = check_array(dictionary, order="F", copy=False)
        self.D = np.require(dictionary, requirements="W")
        # Projection on the constraint set ||V_k|| <= 1
        for j in range(n_components):
            self.D[j] /= max(np.linalg.norm(self.D[j]), 1)
        
        self.C = None
        self.alpha = alpha
        self.method = method
        self.archives = None
        self._verbose = verbose
        self._positive_code = positive_code
        
    def get_gates(self, sample: np.ndarray):
        gates = sparse_encode(
            sample,
            self.D,
            algorithm=self.method, 
            alpha=self.alpha,
            check_input=False,
            positive=self._positive_code,
            max_iter=10000)
        
        if self._verbose:
            recon = np.dot(gates, self.D)
            print('Gates Learning Stage')
            print(f'Level of sparsity: {np.mean(gates == 0):.4f}')
            print(f'Recontruction loss: {np.mean((sample - recon)**2):.4e}')
            print('----------------------------------')

        return gates

    def update_dict(self, codes: np.ndarray, sample: np.ndarray):
        self.N += 1
        # recording
        if self.C is None:
            self.C = codes
            self.archives = sample
        else:
            self.C = np.vstack([self.C, codes])
            self.archives = np.vstack([self.archives, sample])
        assert self.C.shape[0] == self.N
            
        # Update the auxiliary variables
        batch_size = 1
        if self.N < batch_size - 1:
            theta = float((self.N + 1) * batch_size)
        else:
            theta = float(batch_size**2 + self.N + 1 - batch_size)
        beta = (theta + 1 - batch_size) / (theta + 1)

        self.A *= beta
        self.A += np.dot(codes.T, codes)
        self.B *= beta
        self.B += np.dot(sample.T, codes)

        # pre-verbose
        if self._verbose:
            recons = np.dot(self.C, self.D)
            print('Dicts Learning Stage')
            print(f'Pre-recontruction loss: {np.mean((self.archives - recons)**2):.4e}')
        # Update dictionary
        self.D = _update_dict(
            self.D,
            sample,
            codes,
            self.A,
            self.B,
            verbose=self._verbose,
            random_state=self.rng,
            positive=False,
        )
        # post-verbose
        if self._verbose:
            recons = np.dot(self.C, self.D)
            print(f'Post-recontruction loss: {np.mean((self.archives - recons)**2):.4e}')
            print('----------------------------------')

    def _compute_sim_matrix(self, mat: np.ndarray):
        assert np.all(self.C.shape == mat.shape)
        dist_matrix = spatial.distance_matrix(self.C, mat)

        # compute similarity matrix
        dist_matrix -= np.min(dist_matrix)
        dist_matrix /= np.max(dist_matrix)
        simi_matrix = 1 - dist_matrix

        return simi_matrix

    def _compute_corr_matrix(self):
        dist_matrix = spatial.distance_matrix(self.C, self.C)

        # compute correlation matrix
        dist_matrix -= np.min(dist_matrix)
        dist_matrix /= np.max(dist_matrix)
        corr_matrix = 1 - dist_matrix

        return corr_matrix


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L12-v2')

    # task hints
    hints = [
        'Hammer a screw on the wall.',
        'Bypass a wall and push a puck to a goal.',
        'Rotate the faucet clockwise.',
        'Pull a puck to a goal.',
        'Grasp a stick and pull a box with the stick.',
        'Press a handle down sideways.',
        'Push the puck to a goal.',
        'Pick and place a puck onto a shelf.',
        'Push and close a window.',
        'Unplug a peg sideways.',
        # 'Hammer a screw on the wall.',
        # 'Bypass a wall and push a puck to a goal.',
        # 'Rotate the faucet clockwise.',
        # 'Pull a puck to a goal.',
        # 'Grasp a stick and pull a box with the stick.',
        # 'Press a handle down sideways.',
        # 'Push the puck to a goal.',
        # 'Pick and place a puck onto a shelf.',
        # 'Push and close a window.',
        # 'Unplug a peg sideways.'
    ]
    task_idx = [
        'task 0', 'task 1', 'task 2', 'task 3', 'task 4',
        'task 5', 'task 6', 'task 7', 'task 8', 'task 9',
        # 'task 0', 'task 1', 'task 2', 'task 3', 'task 4',
        # 'task 5', 'task 6', 'task 7', 'task 8', 'task 9'
    ]

    dict_learner = OnlineDictLearnerV2(
        n_features=384,
        n_components=1024,
        seed=2,
        alpha=1e-3,
        method='lasso_lars',
        positive_code=False,
        verbose=True)

    # mimic training stage
    for idx, hint_task in enumerate(hints):
        print(idx+1, hint_task)
        task_embedding = model.encode(hint_task)

        # compute gates for current task
        gates = dict_learner.get_gates(task_embedding[np.newaxis, :])
        
        if idx < 10:
            # mimic RL finetuning
            gates += np.random.normal(size=gates.shape) * 0.01

        # online update dictionary via CD
        dict_learner.update_dict(gates, task_embedding[np.newaxis, :])

    # mimic testing stage
    # res_gates = []
    # for idx, hint_task in enumerate(hints):
    #     print(idx+1, hint_task)
    #     task_embedding = model.encode(hint_task)
    #     # compute gates for current task
    #     gates = dict_learner.get_gates(task_embedding[np.newaxis, :])
    #     res_gates.append(gates)
    # testing_gates = np.vstack(res_gates)

    # plot similarity / correlation
    # mat = dict_learner._compute_sim_matrix(testing_gates)
    mat = dict_learner._compute_corr_matrix()

    fig, ax = plt.subplots(figsize=(7, 5)) # 10: (7, 5), 20: (12, 8)

    im, cbar = heatmap(mat, task_idx, task_idx, ax=ax,
                    cmap="YlGn", font_label={'size': 12},
                    x_label='', y_label='', 
                    cbarlabel="Similarity")
    texts = annotate_heatmap(im, valfmt="{x:.1f}")

    fig.tight_layout()
    plt.savefig('corr_plot_10.pdf', bbox_inches='tight', pad_inches=0.02)
    