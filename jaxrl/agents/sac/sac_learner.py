"""Implementations of algorithms for continuous control."""

from copy import deepcopy
import functools
from typing import Optional, Sequence, Tuple, Callable, Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import optax
import wandb
from flax import linen as nn
from flax.core import freeze, unfreeze, FrozenDict

import jaxrl.networks.common as utils_fn
from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.actor import update as update_actor
from jaxrl.agents.sac.critic import target_update
from jaxrl.agents.sac.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey, Params, \
    ModelActor, global_norm, set_optimizer, default_init, tree_l1_mean, \
    replace_embeds
from jaxrl.dict_learning.task_dict import OnlineDictLearner


@functools.partial(jax.jit, static_argnames=('backup_entropy', 'update_target'))
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float,
    target_entropy: float, backup_entropy: bool, update_target: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            backup_entropy=backup_entropy)
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


class SACLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 optim_configs: dict = {},
                 hidden_dims: Sequence[int] = (256, 256),
                 name_activation: str = 'leaky_relu',
                 use_layer_norm: bool = True,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 backup_entropy: bool = True,
                 init_temperature: float = 1.0,
                 init_mean: Optional[np.ndarray] = None,
                 policy_final_fc_init_scale: float = 1.0):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy
        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        actor_def = policies.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            name_activation=name_activation,
            use_layer_norm=use_layer_norm,
            init_mean=init_mean,
            final_fc_init_scale=policy_final_fc_init_scale)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=set_optimizer(**optim_configs))

        critic_def = critic_net.DoubleCritic(
            hidden_dims=hidden_dims,
            name_activation='leaky_relu',
            use_layer_norm=use_layer_norm)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=set_optimizer(**optim_configs))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=set_optimizer(3e-4, 0.01, 'adam', 'global_clip'))

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng

        # for reset models
        self.dummy_a = actions
        self.dummy_o = observations
        self.hidden_dims = hidden_dims
        self.name_activation = name_activation
        self.use_layer_norm = use_layer_norm
        self.init_temp = init_temperature
        self.final_fc_init_scale = policy_final_fc_init_scale

        self.step = 0

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1).flatten()

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1

        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy,
            self.backup_entropy, self.step % self.target_update_period == 0)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info

    def reset_models(self):
        self.actor = self.actor.reset_optimizer()
        self.critic = self.critic.reset_optimizer()
        self.temp = self.temp.reset_optimizer()


@functools.partial(jax.jit, static_argnames=('backup_entropy'))
def _learn_codes_jit(
    rng: PRNGKey,
    actor: ModelActor, 
    critic: Model, 
    target_critic: Model,
    temp: Model, 
    batch: Batch, 
    discount: float, 
    tau: float,
    target_entropy: float,
    backup_entropy: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            backup_entropy=backup_entropy)
    new_target_critic = target_update(new_critic, target_critic, tau)

    rng, key = jax.random.split(rng)
    new_coder, coder_info = update_coder(key, actor, new_critic, temp, batch)
    
    new_temp, alpha_info = temperature.update(temp, coder_info['entropy'],
                                              target_entropy)

    return rng, new_coder, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **coder_info,
        **alpha_info
    }


def update_coder(
    key: PRNGKey, actor: ModelActor, critic: Model, temp: Model, 
    batch: Batch) -> Tuple[Model, InfoDict]:

    def coder_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({'params': actor_params}, batch.observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        coder_loss = (log_probs * temp() - q).mean() #+ 1e-5 * jnp.sum(jnp.abs(codes))
        return coder_loss, {
            'coder_loss': coder_loss,
            'entropy': -log_probs.mean(),
            'L1_codes': jnp.sum(jnp.abs(actor_params['codes'])),
            'L1_components': jnp.sum(jnp.abs(actor_params['components']))}

    new_actor, info = actor.finetune(coder_loss_fn)
    return new_actor, info


# class TaskCoder(nn.Module):
#     initial_codes: jnp.ndarray

#     @nn.compact
#     def __call__(self) -> jnp.ndarray:
#         codes = self.param('codes', init_fn=lambda key: self.initial_codes)
#         return codes


@functools.partial(jax.jit, static_argnames=('backup_entropy'))
def _learn_actor_jit(
    rng: PRNGKey, actor: ModelActor, critic: Model, target_critic: Model, temp: Model, 
    batch: Batch, discount: float, tau: float, target_entropy: float, 
    backup_entropy: bool) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            backup_entropy=backup_entropy)
    new_target_critic = target_update(new_critic, target_critic, tau)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, 
                                         actor, 
                                         new_critic, 
                                         temp, 
                                         batch)

    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


@functools.partial(jax.jit, static_argnames=('backup_entropy', 'with_task_embed'))
def _learn_actor_lwf_jit(
    rng: PRNGKey, embeds: jnp.ndarray, components: jnp.ndarray, actor: Model, 
    critic: Model, target_critic: Model, temp: Model, follower_params: Params,
    batch: Batch, discount: float, tau: float, target_entropy: float, 
    backup_entropy: bool, with_task_embed: bool=True
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            backup_entropy=backup_entropy,
                                            with_task_embedding=with_task_embed,
                                            embeds=embeds)
    new_target_critic = target_update(new_critic, target_critic, tau)

    rng, key = jax.random.split(rng)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        def component_loss(x):
            dist = actor.apply_fn(
                {'params': actor_params}, batch.observations, x.reshape(1, -1))
            dist_targ = actor.apply_fn(
                {'params': follower_params}, batch.observations, x.reshape(1, -1))
            # sampling actions
            actions = dist.sample(seed=key)
            actions_targ = jax.lax.stop_gradient(dist_targ.sample(seed=key))
            return jnp.mean((actions - actions_targ)**2)

        # LwF loss for each component:
        losses = jax.vmap(component_loss)(components)

        # SAC loss
        dist = actor.apply_fn({'params': actor_params}, batch.observations, embeds)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()

        total_loss = actor_loss + 1000 * jnp.sum(losses)
        return total_loss, {
            'sac_pi_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'lwf_pi_loss': jnp.sum(losses)
        }

    new_actor, actor_info = actor.apply_gradient(actor_loss_fn)

    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


class SACTaskDescriptor(SACLearner):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        task_codes: jnp.ndarray,
        task_components: jnp.ndarray,
        coder_lr: float = 1e-3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        name_activation: str = 'leaky_relu',
        use_layer_norm: bool = True,
        discount: float = 0.99,
        tau: float = 0.005,
        target_update_period: int = 1,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        init_temperature: float = 1.0,
        init_mean: Optional[jnp.ndarray] = None,
        policy_final_fc_init_scale: float = 1.0):
        super(SACTaskDescriptor, self).__init__(seed, observations, actions, actor_lr, \
            critic_lr, temp_lr, hidden_dims, name_activation, use_layer_norm, discount,\
            tau, target_update_period, target_entropy, backup_entropy, init_temperature,\
            init_mean, policy_final_fc_init_scale)
        
        action_dim = actions.shape[-1]
        self.coder_lr = coder_lr
        # embeddings = jax.lax.dot(task_codes, task_components)

        # redefine actor
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, coder_key = jax.random.split(rng, 3)
        # actor_def = policies.ConditionalNTPolicy(
        #     hidden_dims,
        #     action_dim,
        #     name_activation=name_activation,
        #     use_layer_norm=use_layer_norm,
        #     init_mean=init_mean,
        #     final_fc_init_scale=policy_final_fc_init_scale)

        actor_def = policies.EmbeddedNTPolicy(
            hidden_dims,
            action_dim,
            task_codes.shape,
            task_components.shape,
            name_activation=name_activation,
            use_layer_norm=use_layer_norm,
            init_mean=init_mean,
            final_fc_init_scale=policy_final_fc_init_scale)
        
        self.actor = ModelActor.create(
            actor_def,
            inputs=[actor_key, observations],
            components=task_components,
            tx=optax.adam(learning_rate=self.actor_lr),
            ftd_tx=optax.adam(learning_rate=coder_lr))

        # self.actor = Model.create(
        #     actor_def,
        #     inputs=[actor_key, observations],
        #     tx=optax.adam(learning_rate=self.actor_lr)
        # )

        # print(jax.jit(self.actor.apply_fn)({'params': self.actor.params}, observations))
        # breakpoint()
        # define previous actor
        self.follower_params = None

        # define task codes
        # self.coder = Model.create(
        #     TaskCoder(task_codes),
        #     inputs=[coder_key],
        #     tx=optax.adam(learning_rate=coder_lr))
        # coder = TaskCoder(task_codes)
        # variables = coder.init(coder_key)
        # _, params = variables.pop('params')


        # print(jax.jit(self.actor.apply_fn)({'params': self.actor.params}, observations))
        # print(self.actor.params['codes'])
        # print(self.actor.params['components'])
        # breakpoint()

        self.rng = rng

    def learn_codes(self, batch: Batch) -> InfoDict:
        self.step += 1

        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _learn_codes_jit(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.backup_entropy)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info

    def learn_actor(self, batch: Batch) -> InfoDict:
        self.step += 1

        # if self.follower_params is None:
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _learn_actor_jit(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.backup_entropy)
        # else:
        #     new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _learn_actor_lwf_jit(
        #         self.rng,
        #         self.embeds,
        #         self.task_components,
        #         self.actor,
        #         self.critic,
        #         self.target_critic,
        #         self.temp,
        #         self.follower_params,
        #         batch,
        #         self.discount,
        #         self.tau,
        #         self.target_entropy,
        #         self.backup_entropy)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info

    def sample_a_with_task_embed(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1).flatten()

    def get_task_embedding(self):
        unfreeze_params = self.actor.params.unfreeze()
        return jnp.dot(unfreeze_params['codes'], unfreeze_params['components'])

    def update_codes_compos(self, new_codes: jnp.ndarray, new_components: jnp.ndarray):
        # self.task_components = new_components
        # coder_def = TaskCoder(new_codes)
        # self.rng, key_coder = jax.random.split(self.rng)
        # variables = coder_def.init(key_coder)
        # _, new_params = variables.pop('params')
        # self.coder = self.coder.update_params(new_params)
        # self.embeds = jax.lax.dot(self.coder(), new_components)

        unfreeze_params = self.actor.params.unfreeze()
        unfreeze_params['codes'] = new_codes
        unfreeze_params['components'] = new_components
        self.actor.update_params(unfreeze_params)

    def reset_models_except_pi(self):
        # reset params for q, q_targ and temp
        self.rng, key_critic, key_temp = jax.random.split(self.rng, 3)
        critic_def = critic_net.DoubleCritic(
            self.hidden_dims,
            name_activation=self.name_activation,
            use_layer_norm=self.use_layer_norm)
        _, new_params_critic = critic_def.init(
            key_critic, self._holder_o, self._holder_a).pop('params')
        self.critic = self.critic.update_params(new_params_critic)
        self.target_critic = self.target_critic.update_params(new_params_critic)

        temp_def = temperature.Temperature(self.init_temp)
        _, new_params_temp = temp_def.init(key_temp).pop('params')
        self.temp = self.temp.update_params(new_params_temp)

        # save the current actor params for LwF
        self.follower_params = deepcopy(self.actor.params)

    def reset_optimizers(self):
        # self.coder = self.coder.reset_optimizer(optax.adam(self.coder_lr))
        self.actor = self.actor.reset_optimizer(
            optax.adam(learning_rate=self.actor_lr),
            optax.adam(learning_rate=self.coder_lr))
        self.critic = self.critic.reset_optimizer(optax.adam(self.critic_lr))
        self.temp = self.temp.reset_optimizer(optax.adam(self.temp_lr))


@functools.partial(jax.jit, static_argnames=('backup_entropy', 'task_i'))
def _update_hat_jit(
    rng: PRNGKey, task_i: int, s_max: float, s: float, cum_mask: Params, pms_mask: Params, 
    emb_mask: FrozenDict, thres_emb: float, alpha: float, actor: Model, critic: Model, 
    target_critic: Model, temp: Model, batch: Batch, discount: float, tau: float,
    target_entropy: float, backup_entropy: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    dist, _ = actor(batch.next_observations, jnp.array([task_i]), s)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)
    target_q = batch.rewards + discount * batch.masks * next_q
    if backup_entropy:
        target_q -= discount * batch.masks * temp() * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }
    
    new_critic, critic_info = critic.apply_gradient(critic_loss_fn)
    new_target_critic = target_update(new_critic, target_critic, tau)

    rng, key = jax.random.split(rng)
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, masks = actor.apply_fn({'params': actor_params}, batch.observations, jnp.array([task_i]), s)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()
        # masking regularization
        reg = 0
        count = 0
        if task_i > 0:
            for m_k, pm_k in zip(masks.keys(), cum_mask.keys()):
                aux = 1 - cum_mask[pm_k]['embedding']
                reg += (masks[m_k]['embedding'] * aux).sum()
                count += aux.sum()
        else:
            for m_k in masks.keys():
                reg += masks[m_k]['embedding'].sum()
                count += masks[m_k]['embedding'].size
        reg /= count
        actor_loss = actor_loss + alpha * reg

        flatten_params, _ = ravel_pytree(actor_params)
        flatten_masks, _ = ravel_pytree(masks)
        return actor_loss, {
            'hac_sac_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'reg_sparsity': reg,
            'params_hist': jnp.histogram(flatten_params, bins=20),
            'masks_hist': jnp.histogram(flatten_masks, bins=20)}
    
    grads_actor, actor_info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    grads_actor = unfreeze(grads_actor)
    # Restrict layer gradients in backprop
    if task_i > 0:
        flatten_pms_mask, _ = ravel_pytree(pms_mask)
        actor_info['pms_mask_hist'] = jnp.histogram(flatten_pms_mask, bins=20)

        for k in pms_mask.keys():
            for sub_k in pms_mask[k].keys():
                grads_actor[k][sub_k] *= pms_mask[k][sub_k]
    # Compensate embedding gradients should be
    # no useful for adaptive optimizers such as Adam
    # according to the original paper's appendix
    # for k in emb_mask.keys():
    #     data = grads_actor[k]['embedding'][task_i]
    #     num = jnp.cosh(jnp.clip(s*data, -50.0, 50.0)) + 1
    #     den = jnp.cosh(data) + 1
    #     data = data * s_max / s * num / den
    #     grads_actor[k]['embedding'] = grads_actor[k]['embedding'].at[task_i].set(data)
    
    # grads monitor
    flatten_grad_actor, _ = ravel_pytree(grads_actor)
    actor_info['actor_grads_hist'] = jnp.histogram(flatten_grad_actor, bins=20)

    new_actor = actor.apply_gradient(grads=freeze(grads_actor), has_aux=False)
    # Constrain embeddings
    params_actor = unfreeze(new_actor.params)
    for k in emb_mask.keys():
        params_actor[k]['embedding'] = jnp.clip(
            params_actor[k]['embedding'], -thres_emb, thres_emb)
    new_actor = new_actor.replace(params=freeze(params_actor))

    # new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
    #                                           target_entropy)

    # fixed temp
    new_temp = temp
    alpha_info = {'temperature': temp(), 'temp_loss': -69}

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }

@functools.partial(jax.jit, static_argnames=('actor_apply_fn'))
def _sample_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: np.ndarray,
        task_i: jnp.ndarray,
        s: float,
        temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray, dict]:
    rng, key = jax.random.split(rng)
    dist, dicts = actor_apply_fn(
        {'params': actor_params},
        observations,
        task_i, s,
        temperature)

    return rng, dist.sample(seed=key), dicts


class HATSACLearner(SACLearner):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        task_num: int,
        max_steps: int = 1000,
        start_training: int = 10000,
        thres_emb: float = 6.0,
        alpha: float = 0.01,
        s_max: float = 200.0,
        transition_steps: int = 1,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        name_activation: str = 'leaky_relu',
        use_layer_norm: bool = False,
        discount: float = 0.99,
        tau: float = 0.005,
        target_update_period: int = 1,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        init_temperature: float = 1.0,
        init_mean: Optional[jnp.ndarray] = None,
        policy_final_fc_init_scale: float = 1.0):
        super(HATSACLearner, self).__init__(seed, observations, actions, actor_lr, \
            critic_lr, temp_lr, hidden_dims, name_activation, use_layer_norm, discount,\
            tau, target_update_period, target_entropy, backup_entropy, init_temperature,\
            init_mean, policy_final_fc_init_scale)

        action_dim = actions.shape[-1]

        self.rng, actor_key = jax.random.split(self.rng)
        actor_def = policies.HatSacPolicy(
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            task_num=task_num,
            name_activation=name_activation,
            use_layer_norm=use_layer_norm,
            final_fc_init_scale=policy_final_fc_init_scale)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations, jnp.array([0])],
                             tx=optax.adam(learning_rate=actor_lr))

        # get grads masking func
        def get_grad_masks(model, masks):
            g_masks = model.get_grad_masks(masks)
            return g_masks
        get_grad_masks_jit = jax.jit(nn.apply(get_grad_masks, actor_def))

        # prerecord embeddings dict:
        mask_emb = {}
        for k in actor.params.keys():
            if k.startswith('emb'):
                mask_emb[k] = 0
        
        self.actor = actor
        self.get_grad_masks = get_grad_masks_jit
        self.thres_emb = thres_emb
        self.alpha = alpha          # Grid search = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4]; chosen was 0.75
        self.s_max = s_max          # Grid search = [25, 50, 100, 200, 400, 800]; chosen was 400
        self.s = 1 / s_max
        self.max_steps = max_steps
        self.start_training = start_training
        self.prac_max_steps = (self.max_steps-self.start_training-1) // transition_steps
        self.transition_steps = transition_steps
        self.mask_emb = freeze(mask_emb)
        self.mask_cum = None
        self.mask_prm = None

        # scale factor schedule
        # self.scdrs = [
        #     optax.linear_schedule(
        #         init_value=self.s_max,
        #         end_value=1/self.s_max,
        #         transition_steps=2000),
        #     optax.linear_schedule(
        #         init_value=1/self.s_max,
        #         end_value=self.s_max,
        #         transition_steps=self.max_steps-self.start_training-2000)
        # ]
        # self.s_scdr = optax.join_schedules(self.scdrs, [2000])
        self.s_scdr = optax.linear_schedule(
            init_value=1/self.s_max,
            end_value=self.s_max,
            transition_steps=self.max_steps-self.start_training)

    def sample_actions(self,
                       observations: np.ndarray,
                       task_i: int,
                       temperature: float = 1.0) -> np.ndarray:
        if temperature == 0:
            # evaluation mode
            s_t = self.s_max
        else:
            # exploration mode
            s_t = self.s

        rng, actions, _ = _sample_actions(self.rng, self.actor.apply_fn,
                                              self.actor.params, observations, 
                                              jnp.array([task_i]), s_t, temperature)
        self.rng = rng

        return actions

    def update(self, task_i: int, batch: Batch) -> InfoDict:
        # prac_step = self.step // self.transition_steps
        # self.s = (self.s_max - 1/self.s_max) * prac_step/self.prac_max_steps + 1/self.s_max
        self.s = self.s_scdr(self.step)
        self.step += 1

        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_hat_jit(
            self.rng, task_i, self.s_max, self.s, self.mask_cum, self.mask_prm, self.mask_emb,
            self.thres_emb, self.alpha, self.actor, self.critic, self.target_critic,
            self.temp, batch, self.discount, self.tau, self.target_entropy, self.backup_entropy)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        info['scale_factor'] = self.s
        info['params_hist'] = wandb.Histogram(np_histogram=info['params_hist'])
        info['masks_hist'] = wandb.Histogram(np_histogram=info['masks_hist'])
        info['actor_grads_hist'] = wandb.Histogram(np_histogram=info['actor_grads_hist'])
        if task_i > 0:
            info['pms_mask_hist'] = wandb.Histogram(np_histogram=info['pms_mask_hist'])
        return info

    def end_task(self, task_i: int):
        # reset scaling parameter schedule for next task
        # self.s = 1 / self.s_max
        self.step = 0
        self.s_scdr = optax.linear_schedule(
            init_value=1/self.s_max,
            end_value=self.s_max,
            transition_steps=self.max_steps-self.start_training)

        self.rng, _, mask_now = _sample_actions(self.rng, self.actor.apply_fn,
                                            self.actor.params, self.dummy_o, 
                                            jnp.array([task_i]), self.s_max)

        # cumulative attention from all the previous tasks
        if task_i > 0:
            self.mask_cum = unfreeze(self.mask_cum)
            for k in self.mask_cum.keys():
                self.mask_cum[k]['embedding'] = jnp.maximum(
                    self.mask_cum[k]['embedding'], 
                    mask_now[k]['embedding'])
        else:
            self.mask_cum = deepcopy(mask_now)
        self.mask_cum = freeze(self.mask_cum)

        # To condition the training of task t + 1, we compute 
        # the gradient masks according to cumulative attention weights
        grad_masks = self.get_grad_masks(
            {'params': self.actor.params}, self.mask_cum)
        grad_masks = jax.tree_util.tree_map(
            lambda x: jnp.where(x < 1.0, 0.0, x), grad_masks)
        self.mask_prm = freeze(grad_masks)

        # reset params for critic ,target_critic and temp
        self.rng, key_critic, key_temp, key_actor = jax.random.split(self.rng, 4)
        critic_def = critic_net.DoubleCritic(
            self.hidden_dims,
            name_activation=self.name_activation,
            use_layer_norm=self.use_layer_norm)
        temp_def = temperature.Temperature(self.init_temp)

        _, new_params_critic = critic_def.init(
            key_critic, self.dummy_o, self.dummy_a).pop('params')
        _, new_params_temp = temp_def.init(key_temp).pop('params')

        self.critic = self.critic.update_params(new_params_critic)
        self.target_critic = self.target_critic.update_params(new_params_critic)
        self.temp = self.temp.update_params(new_params_temp)

        # reset log_std_layer params
        params_actor = unfreeze(self.actor.params)
        log_std = params_actor['log_std_layer']['kernel']
        init_log_std = jax.nn.initializers.orthogonal(scale=self.final_fc_init_scale)(
            key_actor, log_std.shape, jnp.float32)
        params_actor['log_std_layer']['kernel'] = init_log_std
        self.actor.update_params(freeze(params_actor))


@functools.partial(jax.jit, static_argnames=('backup_entropy', 'finetune', 'first_task'))
def _update_spc_jit(
    rng: PRNGKey, task_id: int, coder: Model, s_end: float, s: float, cum_mask: Params, 
    pms_mask: Params, comp_emb: FrozenDict, thres_emb: float, alpha: float, actor: Model, 
    critic: Model, target_critic: Model, temp: Model, batch: Batch, discount: float, tau: float,
    target_entropy: float, backup_entropy: bool, finetune: bool, first_task: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    def coder_loss_fn(coder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        codes = coder.apply_fn({'params': coder_params})
        new_actor_params = replace_embeds(actor.params, codes, comp_emb, task_id)
        dist, dicts = actor.apply_fn({'params': new_actor_params}, batch.observations, jnp.array([task_id]), s)
        masks = dicts['masks']
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()
        # masking regularization
        reg = 0.0
        count = 0
        if first_task:
            for m_k in masks.keys():
                reg += masks[m_k]['embedding'].sum()
                count += masks[m_k]['embedding'].size
        else:
            for m_k, pm_k in zip(masks.keys(), cum_mask.keys()):
                aux = 1 - cum_mask[pm_k]['embedding']
                reg += (masks[m_k]['embedding'] * aux).sum()
                count += aux.sum()
        reg /= count
        actor_loss += alpha * reg

        return actor_loss, {
            'hac_sac_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'reg_sparsity': reg,
            'means': dicts['means'].mean()}

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, dicts = actor.apply_fn({'params': actor_params}, batch.observations, jnp.array([task_id]), s)
        masks = dicts['masks']
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()
        # masking regularization
        reg = 0
        count = 0
        if first_task:
            for m_k in masks.keys():
                reg += masks[m_k]['embedding'].sum()
                count += masks[m_k]['embedding'].size
        else:
            for m_k, pm_k in zip(masks.keys(), cum_mask.keys()):
                aux = 1 - cum_mask[pm_k]['embedding']
                reg += (masks[m_k]['embedding'] * aux).sum()
                count += aux.sum()
        reg /= count
        actor_loss += alpha * reg

        return actor_loss, {
            'hac_sac_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'reg_sparsity': reg,
            'means': dicts['means'].mean()}
    
    if finetune:
        grads_coder, actor_info = jax.grad(coder_loss_fn, has_aux=True)(coder.params)
        new_coder = coder.apply_gradient(grads=grads_coder, has_aux=False)
        codes = new_coder()
        new_actor_params = replace_embeds(actor.params, codes, comp_emb, task_id)
        new_actor = actor.replace(params=new_actor_params)
        # grads norm
        actor_info['g_norm_actor'] = global_norm(grads_coder)
        if not first_task:
            actor_info['used_capacity'] = 1.0 - tree_l1_mean(pms_mask)
        else:
            actor_info['used_capacity'] = 0.0
    else:
        grads_actor, actor_info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
        grads_actor = unfreeze(grads_actor)
        # Restrict layer gradients in backprop
        if not first_task:
            for k in pms_mask.keys():
                for sub_k in pms_mask[k].keys():
                    grads_actor[k][sub_k] *= pms_mask[k][sub_k]
            actor_info['used_capacity'] = 1.0 - tree_l1_mean(pms_mask)
        else:
            actor_info['used_capacity'] = 0.0
        # Compensate embedding gradients should be
        # no useful for adaptive optimizers such as Adam
        # according to the original paper's appendix
        # for k in comp_emb.keys():
        #     data = grads_actor[k]['embedding'][task_id]
        #     num = jnp.cosh(jnp.clip(s*data, -50.0, 50.0)) + 1
        #     den = jnp.cosh(data) + 1
        #     data = data * s_end / s * num / den
        #     grads_actor[k]['embedding'] = grads_actor[k]['embedding'].at[task_id].set(data)

        # grads norm
        actor_info['g_norm_actor'] = global_norm(grads_actor)
        new_actor = actor.apply_gradient(grads=freeze(grads_actor), has_aux=False)
        # Constrain embeddings
        actor_params = unfreeze(new_actor.params)
        for k in comp_emb.keys():
            actor_params[k]['embedding'] = jnp.clip(
                actor_params[k]['embedding'], -thres_emb, thres_emb)
        new_actor = new_actor.replace(params=freeze(actor_params))
        new_coder = coder

    # monitoring embeddings
    in_embeds = {}
    for k in comp_emb.keys():
        in_embeds[k] = new_actor.params[k]['embedding'][task_id]
    actor_info['l1_mean_embeds'] = tree_l1_mean(in_embeds)
    actor_info['l1_mean_codes'] = tree_l1_mean(new_coder())

    def temperature_loss_fn(temp_params: Params):
        temperature = temp.apply_fn({'params': temp_params})
        temp_loss = temperature * (actor_info['entropy'] - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}
    
    if finetune:
        new_temp = temp
        alpha_info = {'temperature': temp(), 'temp_loss': -69, 'g_norm_temp': -69}
    else:
        grads_temp, alpha_info = jax.grad(temperature_loss_fn, has_aux=True)(temp.params)
        alpha_info['g_norm_temp'] = global_norm(grads_temp)  
        new_temp = temp.apply_gradient(grads=grads_temp, has_aux=False)
    
    rng, key = jax.random.split(rng)
    dist, _ = actor(batch.next_observations, jnp.array([task_id]), s)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)
    target_q = batch.rewards + discount * batch.masks * next_q
    if backup_entropy:
        target_q -= discount * batch.masks * temp() * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()}
    
    grads_critic, critic_info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    # grads norm
    critic_info['g_norm_critic'] = global_norm(grads_critic)
    new_critic = critic.apply_gradient(grads=grads_critic, has_aux=False)
    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_coder, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info}


class SPCLearner(SACLearner):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        task_nums: int,
        max_steps: int = 1000,
        start_training: int = 10000,
        finetune_steps: int = 10000,
        thres_emb: float = 6.0,
        alpha: float = 0.01,
        s_warm_start: bool = False,
        s_start: float = 400.0,
        s_end: float = 1/400.0,
        component_nums: int = 5,
        optim_configs: dict = {},
        hidden_dims: Sequence[int] = (256, 256),
        name_activation: str = 'leaky_relu',
        use_layer_norm: bool = False,
        use_rms_norm: bool = True,
        discount: float = 0.99,
        tau: float = 0.005,
        target_update_period: int = 1,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        init_temperature: float = 1.0,
        init_mean: Optional[jnp.ndarray] = None,
        policy_final_fc_init_scale: float = 1e-3):
        super(SPCLearner, self).__init__(seed, observations, actions, optim_configs,
              hidden_dims, name_activation, use_layer_norm, discount, tau, target_update_period,
              target_entropy, backup_entropy, init_temperature, init_mean, policy_final_fc_init_scale)

        action_dim = actions.shape[-1]

        self.rng, actor_key, coder_key = jax.random.split(self.rng, 3)
        actor_def = policies.HatTanhPolicy(
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            task_num=task_nums,
            name_activation=name_activation,
            use_layer_norm=use_layer_norm,
            use_rms_norm=use_rms_norm,
            final_fc_init_scale=policy_final_fc_init_scale)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations, jnp.array([0])],
                             tx=set_optimizer(**optim_configs))

        coder_def = policies.Coder(
            hidden_lens=len(hidden_dims),
            compnt_nums=component_nums)
        coder = Model.create(coder_def,
                             inputs=[coder_key],
                             tx=set_optimizer(**optim_configs))

        # get grads masking func
        def get_grad_masks(model, masks):
            g_masks = model.get_grad_masks(masks)
            return g_masks
        get_grad_masks_jit = jax.jit(nn.apply(get_grad_masks, actor_def))

        # prerecord embeddings dict with components for each layer:
        self.dict_component = {}
        comp_embs = {}
        for k in actor.params.keys():
            if k.startswith('emb'):
                comp_embs[k] = {'components': None}
        dict_embs = {}
        for i, k in enumerate(comp_embs.keys()):
            dict_learner = OnlineDictLearner(
                n_features=hidden_dims[i],
                n_components=component_nums,
                seed=seed + i,
                verbose=True)
            dict_embs[k] = {'learner': dict_learner}
            comp_embs[k] = {'components': dict_learner.get_components()}
        # 'lbl': dict_learner by layers
        # 'cbl': component by layers
        self.dict_component['lbl'] = dict_embs
        self.dict_component['cbl'] = comp_embs

        self.s = s_start
        self.actor = actor
        self.coder = coder
        self.alpha = alpha          # Grid search = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4]; chosen was 0.75
        self.s_end = s_end          # Grid search = [25, 50, 100, 200, 400, 800]; chosen was 400
        self.mask_cum = None
        self.mask_prm = None
        self.finetune = True
        self.fir_task = True
        self.s_start = s_start
        self.thres_emb = thres_emb
        self.max_steps = max_steps
        self.s_warm_start = s_warm_start
        self.finetune_steps = finetune_steps
        self.start_training = start_training
        self.component_nums = component_nums
        self.get_grad_masks = get_grad_masks_jit
        
        if self.s_warm_start:
            self.s_scdr = utils_fn.linear_warm_start(
                start_val=self.s_start,
                end_val=self.s_end,
                warm_steps=self.finetune_steps,
                linear_steps=self.max_steps-self.start_training-self.finetune_steps)
        else:
            self.s_scdr = utils_fn.linear_ascent(
                start_val=self.s_start,
                end_val=self.s_end,
                linear_steps=self.max_steps-self.start_training)

    def sample_actions(self,
                       observations: np.ndarray,
                       task_id: int,
                       temperature: float = 1.0) -> jnp.ndarray:
        if temperature == 0:
            # evaluation mode
            s_t = self.s_end
        else:
            # exploration mode
            s_t = self.s

        rng, actions, _ = _sample_actions(self.rng, self.actor.apply_fn,
                                          self.actor.params, observations, 
                                          jnp.array([task_id]), s_t, temperature)
        self.rng = rng
        return actions

    def update(self, task_id: int, batch: Batch) -> InfoDict:
        if self.step >= self.finetune_steps:
            self.finetune = False

        self.step += 1
        self.s = self.s_scdr(self.step)

        new_rng, new_coder, new_actor, new_critic, new_target_critic, new_temp, info = _update_spc_jit(
            rng=self.rng, task_id=task_id, coder=self.coder, s_end=self.s_end, s=self.s, cum_mask=self.mask_cum, 
            pms_mask=self.mask_prm, comp_emb=freeze(self.dict_component['cbl']), thres_emb=self.thres_emb, 
            alpha=self.alpha, actor=self.actor, critic=self.critic, target_critic=self.target_critic, 
            temp=self.temp, batch=batch, discount=self.discount, tau=self.tau, target_entropy=self.target_entropy, 
            backup_entropy=self.backup_entropy, finetune=self.finetune, first_task=self.fir_task)

        self.rng = new_rng
        self.coder = new_coder
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp     

        info['scale_factor'] = self.s
        return info

    def end_task(self, task_id: int):
        # reset scaling parameter schedule for next task
        self.step = 0
        if self.s_warm_start:
            self.s_scdr = utils_fn.linear_warm_start(
                start_val=self.s_start,
                end_val=self.s_end,
                warm_steps=self.finetune_steps,
                linear_steps=self.max_steps-self.start_training-self.finetune_steps)
        else:
            self.s_scdr = utils_fn.linear_ascent(
                start_val=self.s_start,
                end_val=self.s_end,
                linear_steps=self.max_steps-self.start_training)

        self.rng, _, dicts = _sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, self.dummy_o, 
                                             jnp.array([task_id]), self.s_end)
        mask_now = dicts['masks']
        # cumulative attention from all the previous tasks
        if self.fir_task:
            self.mask_cum = deepcopy(mask_now)
        else:
            self.mask_cum = unfreeze(self.mask_cum)
            for k in self.mask_cum.keys():
                self.mask_cum[k]['embedding'] = jnp.maximum(
                    self.mask_cum[k]['embedding'], 
                    mask_now[k]['embedding'])
        self.mask_cum = freeze(self.mask_cum)

        # To condition the training of task t + 1, we compute 
        # the gradient masks according to cumulative attention weights
        grad_masks = self.get_grad_masks(
            {'params': self.actor.params}, self.mask_cum)
        grad_masks = jax.tree_util.tree_map(
            lambda x: jnp.where(x < 1.0, 0.0, 1.0), grad_masks)
        self.mask_prm = freeze(grad_masks)

        # reset params for coder, critic ,target_critic and temperature
        self.rng, key_critic, key_temp, key_actor, key_coder = jax.random.split(self.rng, 5)
        coder_def = policies.Coder(
            hidden_lens=len(self.hidden_dims),
            compnt_nums=self.component_nums)
        critic_def = critic_net.DoubleCritic(
            hidden_dims=self.hidden_dims,
            name_activation='leaky_relu',
            use_layer_norm=self.use_layer_norm)
        temp_def = temperature.Temperature(self.init_temp)

        _, new_params_coder = coder_def.init(key_coder).pop('params')
        _, new_params_critic = critic_def.init(
            key_critic, self.dummy_o, self.dummy_a).pop('params')
        _, new_params_temp = temp_def.init(key_temp).pop('params')

        self.coder = self.coder.update_params(new_params_coder)
        self.critic = self.critic.update_params(new_params_critic)
        self.target_critic = self.target_critic.update_params(new_params_critic)
        self.temp = self.temp.update_params(new_params_temp)

        # reset log_std_layer params
        new_params_actor = utils_fn.reset_logstd_layer(
            key_actor,
            self.actor.params,
            self.final_fc_init_scale)
        self.actor.update_params(new_params_actor)

        # update dictionary learners
        params_actor = self.actor.params
        comp_embs = self.dict_component['cbl']
        dict_embs = self.dict_component['lbl']
        for k in comp_embs.keys():
            embeds = params_actor[k]['embedding'][task_id]
            embeds = np.array([embeds])
            dict_learner = dict_embs[k]['learner']
            dict_learner.decompose(embeds)
            dict_embs[k] = {'learner': dict_learner}
            comp_embs[k] = {'components': jnp.asarray(dict_learner.get_components())}
        self.dict_component['lbl'] = dict_embs
        self.dict_component['cbl'] = comp_embs

        # reset optimizers
        self.coder = self.coder.reset_optimizer()
        self.actor = self.actor.reset_optimizer()
        self.critic = self.critic.reset_optimizer()
        self.temp = self.temp.reset_optimizer()

        self.finetune = True  # finetune again for the next task
        self.fir_task = False # trigger for the next task
