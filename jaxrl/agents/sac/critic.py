from typing import Tuple, Optional

import jax
import optax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, TrainState, Params, PRNGKey


def target_update(critic: TrainState, target_critic: TrainState, tau: float) -> TrainState:
    # new_target_params = jax.tree_multimap(
    #     lambda p, tp: p * tau + tp * (1 - tau), critic.params,
    #     target_critic.params)
    # use optax's implementation
    new_target_params = optax.incremental_update(
        critic.params, target_critic.params, tau
    )
    return target_critic.replace(params=new_target_params)


def update(key: PRNGKey, actor: TrainState, critic: TrainState, target_critic: TrainState,
           temp: TrainState, batch: Batch, discount: float) -> Tuple[TrainState, InfoDict]:

    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)
    next_q -= temp() * next_log_probs
    target_q = batch.rewards + discount * batch.masks * next_q

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    grads_critic, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads_critic)

    return new_critic, info
