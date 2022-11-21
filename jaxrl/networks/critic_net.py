"""Implementations of algorithms for continuous control."""

from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import MLP, activation_fn


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    name_activation: str = 'leaky_relu'
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP(
            (*self.hidden_dims, 1), 
            activations=activation_fn(self.name_activation),
            use_layer_norm=self.use_layer_norm,
            activate_final=False)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    name_activation: str = 'leaky_relu'
    use_layer_norm: bool = True
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions):

        VmapCritic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims,
                        name_activation=self.name_activation,
                        use_layer_norm=self.use_layer_norm)(states, actions)
        return qs
