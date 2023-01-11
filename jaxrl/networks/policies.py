import functools
from typing import Any, Callable, Optional, Sequence, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from flax.linen.module import init
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from jaxrl.networks.common import MLP, Params, PRNGKey, \
    default_init, activation_fn, RMSNorm

# from common import MLP, Params, PRNGKey, default_init, \
#     activation_fn, RMSNorm, create_mask, zero_grads

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class MSEPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> jnp.ndarray:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        actions = nn.Dense(self.action_dim,
                           kernel_init=default_init())(outputs)
        return nn.tanh(actions)


class TanhTransformedDistribution(tfd.TransformedDistribution):
    """Distribution followed by tanh."""

    def __init__(self, distribution, threshold=.999, validate_args=False):
        """Initialize the distribution.
        Args:
          distribution: The distribution to transform.
          threshold: Clipping value of the action when computing the logprob.
          validate_args: Passed to super class.
        """
        super().__init__(
            distribution=distribution,
            bijector=tfb.Tanh(),
            validate_args=validate_args)
        # Computes the log of the average probability distribution outside the
        # clipping range, i.e. on the interval [-inf, -atanh(threshold)] for
        # log_prob_left and [atanh(threshold), inf] for log_prob_right.
        self._threshold = threshold
        inverse_threshold = self.bijector.inverse(threshold)
        # average(pdf) = p/epsilon
        # So log(average(pdf)) = log(p) - log(epsilon)
        log_epsilon = jnp.log(1. - threshold)
        # Those 2 values are differentiable w.r.t. model parameters, such that the
        # gradient is defined everywhere.
        self._log_prob_left = self.distribution.log_cdf(
            -inverse_threshold) - log_epsilon
        self._log_prob_right = self.distribution.log_survival_function(
            inverse_threshold) - log_epsilon

    def log_prob(self, event):
        # Without this clip there would be NaNs in the inner tf.where and that
        # causes issues for some reasons.
        event = jnp.clip(event, -self._threshold, self._threshold)
        # The inverse image of {threshold} is the interval [atanh(threshold), inf]
        # which has a probability of "log_prob_right" under the given distribution.
        return jnp.where(
            event <= -self._threshold, self._log_prob_left,
            jnp.where(event >= self._threshold, self._log_prob_right,
                      super().log_prob(event)))

    def mode(self):
        return self.bijector.forward(self.distribution.mode())

    def entropy(self, seed=None):
        # We return an estimation using a single sample of the log_det_jacobian.
        # We can still do some backpropagation with this estimate.
        return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
            self.distribution.sample(seed=seed), event_ndims=0)

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype,
                                                      num_classes=num_classes)
        del td_properties['bijector']
        return td_properties


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    name_activation: str = 'leaky_relu'
    use_rms_norm: bool = False
    use_layer_norm: bool = False
    state_dependent_std: bool = True
    final_fc_init_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    dropout_rate: Optional[float] = None
    init_mean: Optional[jnp.ndarray] = None
    clip_mean: float = 1.0
    tanh_squash: bool = True

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activations=activation_fn(self.name_activation),
                      activate_final=True,
                      use_layer_norm=self.use_layer_norm)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim,
                         kernel_init=default_init(
                         self.final_fc_init_scale))(outputs)
        if self.init_mean is not None:
            means += self.init_mean

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                    self.final_fc_init_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim,))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        if self.tanh_squash:
            return tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        else:
            return base_dist


class ConditionalNTPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    name_activation: str = 'leaky_relu'
    use_layer_norm: bool = True
    dropout_rate: Optional[float] = None
    final_fc_init_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    init_mean: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 conditions: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        # concatenate the conditions and obs
        batch_s = observations.shape[0]
        conds_d = conditions.shape[1]
        conditions = jnp.broadcast_to(conditions, (batch_s, conds_d))
        inputs = jax.lax.concatenate([conditions, observations], 1)
        # MLP backbone
        outputs = MLP(self.hidden_dims,
                      activations=activation_fn(self.name_activation),
                      activate_final=True,
                      use_layer_norm=self.use_layer_norm)(inputs,
                                                      training=training)

        means = nn.Dense(self.action_dim,
                         kernel_init=default_init(
                             self.final_fc_init_scale))(outputs)
        if self.init_mean is not None:
            means += self.init_mean

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                    self.final_fc_init_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        else:
            return base_dist


class EmbeddedNTPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    codes_dims: Sequence[int]
    components_dims: Sequence[int]
    state_dependent_std: bool = True
    name_activation: str = 'leaky_relu'
    use_layer_norm: bool = True
    dropout_rate: Optional[float] = None
    final_fc_init_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    init_mean: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        # concatenate the embeddings and obs
        codes = self.param("codes", nn.initializers.zeros, self.codes_dims)
        components = self.param("components", nn.initializers.zeros, self.components_dims)
        embeds = jax.lax.dot(codes, components)
        batch_s = observations.shape[0]
        embed_d = embeds.shape[1]
        embeds = jnp.broadcast_to(embeds, (batch_s, embed_d))
        inputs = jax.lax.concatenate([embeds, observations], 1)
        # MLP backbone
        outputs = MLP(self.hidden_dims,
                      activations=activation_fn(self.name_activation),
                      activate_final=True,
                      use_layer_norm=self.use_layer_norm)(inputs,
                                                      training=training)

        means = nn.Dense(self.action_dim,
                         kernel_init=default_init(
                             self.final_fc_init_scale))(outputs)
        if self.init_mean is not None:
            means += self.init_mean

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                    self.final_fc_init_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        else:
            return base_dist


class Coder(nn.Module):
    hidden_lens: int
    compnt_nums: int

    def setup(self):
        self.codes_bb = nn.Embed(
            self.hidden_lens, 
            self.compnt_nums)
        # embedding_init=jax.nn.initializers.zeros
    
    def __call__(self):
        codes = {}
        for i in range(self.hidden_lens):
            codes[f'embeds_bb_{i}'] = {'codes': self.codes_bb(jnp.array([i]))}
        return codes


class MetaPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    task_num: int
    state_dependent_std: bool = True
    name_activation: str = 'leaky_relu'
    use_layer_norm: bool = False
    use_rms_norm: bool = True
    final_fc_init_scale: float = 1.0
    clip_mean: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash: bool = True

    def setup(self):
        if self.use_layer_norm and self.use_rms_norm:
            raise ValueError("use_layer_norm and use_rms_norm cannot both be true")

        self.backbones = [nn.Dense(hidn, kernel_init=default_init()) \
            for hidn in self.hidden_dims]
        self.embeds_bb = [nn.Embed(self.task_num, hidn, embedding_init=default_init(1.0)) \
            for hidn in self.hidden_dims]
        
        self.mean_layer = nn.Dense(
            self.action_dim, 
            kernel_init=default_init(self.final_fc_init_scale),
            use_bias=False)

        if self.state_dependent_std:
            self.log_std_layer = nn.Dense(
                self.action_dim, 
                kernel_init=default_init(self.final_fc_init_scale)
            )
        else:
            self.log_std_layer = self.param(
                'log_std_layer', nn.initializers.zeros, 
                (self.action_dim,)
            )

        self.relu1 = lambda x: jnp.minimum(jnp.maximum(x, 0), 1.)
        self.hard_tanh = lambda x: jnp.where(
            x > self.clip_mean, self.clip_mean, 
            jnp.where(x < -self.clip_mean, -self.clip_mean, x)
        )
        self.activation = activation_fn(self.name_activation)
        if self.use_layer_norm:
            self.normalizer = nn.LayerNorm(use_bias=False, use_scale=False)
        elif self.use_rms_norm:
            self.normalizer = RMSNorm(axis=1)
        else:
            self.normalizer = None

    def __call__(self,
                 x: jnp.ndarray,
                 t: jnp.ndarray,
                 temperature: float = 1.0):
        masks = {}
        for i, layer in enumerate(self.backbones):
            x = layer(x)
            # straight-through estimator
            g = self.relu1(self.embeds_bb[i](t))
            masks[layer.name] = {'embedding': g}
            x = self.activation(x)
            x = x * jnp.broadcast_to(g, x.shape)
            if i == 0 and (self.use_layer_norm or self.use_rms_norm):
                x = self.normalizer(x)
        
        means = self.mean_layer(x)

        # Avoid numerical issues by limiting the mean of the Gaussian
        # to be in [-clip_mean, clip_mean]
        means = self.hard_tanh(means)

        # if not self.tanh_squash:
        # means = nn.tanh(means)

        if self.state_dependent_std:
            log_stds = self.log_std_layer(x)
        else:
            log_stds = self.log_std_layer

        # clip log_std
        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)
        
        # numerically unstable method for unbounded means
        # base_dist = tfd.MultivariateNormalDiag(loc=means,
        #                                        scale_diag=jnp.exp(log_stds) *
        #                                        temperature)

        # numerically stable method
        base_dist = tfd.Normal(loc=means, scale=jnp.exp(log_stds) * temperature)

        if self.tanh_squash:
            return tfd.Independent(TanhTransformedDistribution(base_dist), 
                                   reinterpreted_batch_ndims=1), {'masks': masks, 'means': means}
        else:
            return base_dist, {'masks': masks, 'means': means}

    def get_grad_masks(self, masks: dict, input_dim: int = 12):
        grad_masks = {}
        for i, layer in enumerate(self.backbones):
            if i == 0:
                post_e = masks[layer.name]['embedding']
                grad_masks[layer.name] = {
                    'kernel': 1-jnp.broadcast_to(post_e, (input_dim, self.hidden_dims[i])),
                    'bias': 1-post_e.flatten()}
                pre_e = masks[layer.name]['embedding']
            else:
                post_e = masks[layer.name]['embedding']
                kernel_e = jnp.minimum(
                    jnp.broadcast_to(pre_e.reshape(-1, 1), (self.hidden_dims[i-1], self.hidden_dims[i])),
                    jnp.broadcast_to(post_e, (self.hidden_dims[i-1], self.hidden_dims[i]))
                )
                grad_masks[layer.name] = {'kernel': 1-kernel_e, 'bias': 1-post_e.flatten()}
                pre_e = masks[layer.name]['embedding']

        kernel_e = jnp.broadcast_to(pre_e.reshape(-1, 1), (self.hidden_dims[-1], self.action_dim))
        grad_masks[self.mean_layer.name] = {'kernel': 1-kernel_e}
        
        return grad_masks


class MaskedTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    task_num: int
    state_dependent_std: bool = True
    name_activation: str = 'leaky_relu'
    use_layer_norm: bool = False
    use_rms_norm: bool = True
    final_fc_init_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True

    def setup(self):
        if self.use_layer_norm and self.use_rms_norm:
            raise ValueError("use_layer_norm and use_rms_norm cannot both be true")

        self.backbones = [nn.Dense(hidn, kernel_init=default_init()) \
            for hidn in self.hidden_dims]
        self.embeds_bb = [nn.Embed(self.task_num, hidn, embedding_init=nn.initializers.normal(1.0)) \
            for hidn in self.hidden_dims]
        
        self.mean_layer = nn.Dense(
            self.action_dim, 
            kernel_init=default_init(self.final_fc_init_scale),
            use_bias=False
        )

        if self.state_dependent_std:
            self.log_std_layer = nn.Dense(
                self.action_dim, 
                kernel_init=default_init(self.final_fc_init_scale),
                use_bias=False
            )
        else:
            self.log_std_layer = self.param(
                'log_std_layer', nn.initializers.zeros, 
                (self.action_dim,)
            )

        self.activation = activation_fn(self.name_activation)
        if self.use_layer_norm:
            self.normalizer = nn.LayerNorm(use_bias=False, use_scale=False)
        elif self.use_rms_norm:
            self.normalizer = RMSNorm(axis=1)
        else:
            self.normalizer = None

    def __call__(self,
                 key: chex.PRNGKey,
                 x: jnp.ndarray,
                 t: jnp.ndarray,
                 s: float = 1.0,
                 temperature: float = 1.0):
        masks = {}
        for i, layer in enumerate(self.backbones):
            x = layer(x)
            g = tfd.RelaxedBernoulli(
                temperature=s, 
                logits=self.embeds_bb[i](t)/s).sample(seed=key)
            masks[layer.name] = {'embedding': g}
            x = self.activation(x)
            x = x * jnp.broadcast_to(g, x.shape)
            if self.use_layer_norm or self.use_rms_norm:
                x = self.normalizer(x)
        
        means = self.mean_layer(x)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        if self.state_dependent_std:
            log_stds = self.log_std_layer(x)
        else:
            log_stds = self.log_std_layer

        # clip log_std
        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)
        
        # numerically unstable method for unbounded means
        # base_dist = tfd.MultivariateNormalDiag(loc=means,
        #                                        scale_diag=jnp.exp(log_stds) *
        #                                        temperature)

        # numerically stable method
        base_dist = tfd.Normal(loc=means, scale=jnp.exp(log_stds) * temperature)

        if self.tanh_squash_distribution:
            return tfd.Independent(TanhTransformedDistribution(base_dist), 
                                   reinterpreted_batch_ndims=1), {'masks': masks, 'means': means}
        else:
            return base_dist, {'masks': masks, 'means': means}

    def get_grad_masks(self, masks: dict, input_dim: int = 12):
        grad_masks = {}
        for i, layer in enumerate(self.backbones):
            if i == 0:
                post_e = masks[layer.name]['embedding']
                grad_masks[layer.name] = {
                    'kernel': 1-jnp.broadcast_to(post_e, (input_dim, self.hidden_dims[i])),
                    'bias': 1-post_e.flatten()}
                pre_e = masks[layer.name]['embedding']
            else:
                post_e = masks[layer.name]['embedding']
                kernel_e = jnp.minimum(
                    jnp.broadcast_to(pre_e.reshape(-1, 1), (self.hidden_dims[i-1], self.hidden_dims[i])),
                    jnp.broadcast_to(post_e, (self.hidden_dims[i-1], self.hidden_dims[i]))
                )
                grad_masks[layer.name] = {'kernel': 1-kernel_e, 'bias': 1-post_e.flatten()}
                pre_e = masks[layer.name]['embedding']

        kernel_e = jnp.broadcast_to(pre_e.reshape(-1, 1), (self.hidden_dims[-1], self.action_dim))
        grad_masks[self.mean_layer.name] = {'kernel': 1-kernel_e}
        
        return grad_masks


class NormalTanhMixturePolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    num_components: int = 5
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        logits = nn.Dense(self.action_dim * self.num_components,
                          kernel_init=default_init())(outputs)
        means = nn.Dense(self.action_dim * self.num_components,
                         kernel_init=default_init(),
                         bias_init=nn.initializers.normal(stddev=1.0))(outputs)
        log_stds = nn.Dense(self.action_dim * self.num_components,
                            kernel_init=default_init())(outputs)

        shape = list(observations.shape[:-1]) + [-1, self.num_components]
        logits = jnp.reshape(logits, shape)
        mu = jnp.reshape(means, shape)
        log_stds = jnp.reshape(log_stds, shape)

        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        components_distribution = tfd.Normal(loc=mu,
                                             scale=jnp.exp(log_stds) *
                                             temperature)

        base_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=components_distribution)

        dist = tfd.TransformedDistribution(distribution=base_dist,
                                           bijector=tfb.Tanh())

        return tfd.Independent(dist, 1)


@functools.partial(
    jax.jit, static_argnames=('actor_apply_fn', 'distribution'))
def _sample_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob') -> Tuple[PRNGKey, jnp.ndarray]:
    if distribution == 'det':
        return rng, actor_apply_fn({'params': actor_params}, observations,
                                   temperature)
    else:
        dist = actor_apply_fn(
            {'params': actor_params}, observations, temperature)
    
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


def sample_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: jnp.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob') -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_apply_fn, actor_params, observations,
                           temperature, distribution)


if __name__ == "__main__":
    import optax

    actor = MetaPolicy(
        hidden_dims=(256, 256, 256, 256),
        action_dim=4,
        task_num=20,
        state_dependent_std=False,
        use_layer_norm=False,
        use_rms_norm=False)
    
    rng, key = random.split(random.PRNGKey(0))
    variables = actor.init(
        key, jnp.ones((1, 12)), jnp.array([0])
    )
    _, params = variables.pop('params')
    print(params)

    # tx = optax.multi_transform({'train': optax.adam(0.1), 'fix': optax.set_to_zero()},
    #     create_mask(params, ['backbones', 'mean', 'log']))
    # opt_state = tx.init(params)

    # apply_jit = jax.jit(actor.apply)
    # dist, dicts = apply_jit(variables, jnp.ones((3, 12)), jnp.array([0]), 1e-5)

    # def get_grad_masks(actor, masks):
    #     g_masks = actor.get_grad_masks(masks)
    #     return g_masks
    # get_grad_masks_jit = jax.jit(nn.apply(get_grad_masks, actor))
    # grad_masks = get_grad_masks_jit(variables, masks)

    # def loss(params):
    #     dist, _ = actor.apply({'params': params}, jnp.ones((3, 12)), jnp.array([0]))
    #     samples = dist.sample(seed=random.PRNGKey(0))
    #     return jnp.sum(samples)

    # grads = jax.grad(loss)(params)

    # updates, opt_state = tx.update(grads, opt_state, params)
    # new_params = optax.apply_updates(params, updates)

    # compares = jax.tree_util.tree_map(lambda x, y: x == y, params, new_params)
    # print(compares)
    # for k in grad_masks.keys():
    #     print(k)
    #     if k == 'mean_layer':
    #         assert (grads[k]['kernel'].shape == grad_masks[k]['kernel'].shape)
    #     else:
    #         assert (grads[k]['kernel'].shape == grad_masks[k]['kernel'].shape)
    #         assert (grads[k]['bias'].shape == grad_masks[k]['bias'].shape)

    # print(dicts['masks'])

    # @jax.jit
    # def top_k_fn(data):
    #     return jax.lax.top_k(data, k=2)

    # a = jnp.array([[1, 2, 3, 4, 5]])
    # b = jnp.array([[1,2,3,4,5], [5,4,3,2,1]])
    # x, y = top_k_fn(a)
    # print(x)
    # print(jnp.take_along_axis(b, y, axis=1))
