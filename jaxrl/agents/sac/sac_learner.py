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
from sentence_transformers import SentenceTransformer

import jaxrl.networks.common as utils_fn
from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.actor import update as update_actor
from jaxrl.agents.sac.critic import target_update
from jaxrl.agents.sac.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey, Params, \
    AlterTrainableModel, global_norm, set_optimizer, tree_l1_mean
from jaxrl.dict_learning.task_dict import OnlineDictLearnerV2


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
                 actor_configs: dict = {},
                 critic_configs: dict = {},
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 1.0):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -1.0 * action_dim
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        actor_configs['action_dim'] = action_dim
        actor_def = policies.NormalTanhPolicy(**actor_configs)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=set_optimizer(**optim_configs))

        critic_def = critic_net.DoubleCritic(**critic_configs)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=set_optimizer(**optim_configs))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        opt_kwargs_temp = deepcopy(optim_configs)
        # optim_configs['lr'] = 1e-3
        opt_kwargs_temp['max_norm'] = -1.0
        # optim_configs['optim_algo'] = 'sgd'
        opt_kwargs_temp['clip_method'] = None
        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=set_optimizer(**opt_kwargs_temp))

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng

        # for reset models
        self.dummy_a = actions
        self.dummy_o = observations
        self.actor_cfgs = actor_configs
        self.critic_cfgs = critic_configs
        self.init_temp = init_temperature

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
            self.step % self.target_update_period == 0)

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


@functools.partial(jax.jit, static_argnames=('actor_apply_fn'))
def _sample_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: np.ndarray,
        task_i: jnp.ndarray,
        temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray, dict]:
    rng, key = jax.random.split(rng)
    dist, dicts = actor_apply_fn(
        {'params': actor_params},
        observations,
        task_i,
        temperature)

    return rng, dist.sample(seed=key), dicts


@functools.partial(jax.jit, static_argnames=('finetune', 'first_task'))
def _update_spc_jit(
    rng: PRNGKey, task_id: int, pms_mask: FrozenDict, cum_mask: FrozenDict, actor: AlterTrainableModel,
    critic: Model, target_critic: Model, temp: Model, batch: Batch, discount: float, 
    tau: float, target_entropy: float, finetune: bool, first_task: bool
    ) -> Tuple[PRNGKey, AlterTrainableModel, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, dicts = actor.apply_fn({'params': actor_params}, batch.observations, jnp.array([task_id]))
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()

        if finetune:
            # masking regularization
            reg = 0
            count = 0
            if first_task:
                for m_k in dicts['masks']:
                    reg += dicts['masks'][m_k]['embedding'].sum()
                    count += dicts['masks'][m_k]['embedding'].size
            else:
                for m_k in dicts['masks']:
                    aux = 1 - cum_mask[m_k]['embedding']
                    reg += (dicts['masks'][m_k]['embedding'] * aux).sum()
                    count += aux.sum()
            regularizer = reg / count * jax.lax.stop_gradient(0.5 * jnp.sqrt(jnp.abs(q).mean()))

            actor_loss += regularizer

        # Modified Differential Method of Multipliers
        # epi = -target_entropy
        # damp = damping * jax.lax.stop_gradient(epi - log_probs.mean())
        # temperature = temp.apply_fn({'params': temp_params})
        # actor_loss = -q.mean() - (temperature - damp) * (epi - log_probs.mean())

        _info = {
            'hac_sac_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'means': dicts['means'].mean()
        }
        for k in dicts['masks']:
            _info[k+'_rate_act'] = utils_fn.rate_activity(dicts['masks'][k])

        return actor_loss, _info
    
    # grads of actor
    grads_actor, actor_info = jax.grad(actor_loss_fn,has_aux=True)(actor.params)
    # recording grads norm
    actor_info['g_norm_actor'] = global_norm(grads_actor)

    if finetune:
        # only update coefficients
        new_actor = actor.apply_gradient_one(grads=grads_actor, has_aux=False)
        if not first_task:
            actor_info['used_capacity'] = 1.0 - tree_l1_mean(pms_mask)
        else:
            actor_info['used_capacity'] = 0.0
    else:
        # only update actor params and
        # restrict layer gradients in backprop
        if not first_task:
            grads_actor = unfreeze(grads_actor)
            for k in pms_mask.keys():
                for sub_k in pms_mask[k].keys():
                    grads_actor[k][sub_k] *= pms_mask[k][sub_k]
            grads_actor = freeze(grads_actor)
            actor_info['used_capacity'] = 1.0 - tree_l1_mean(pms_mask)
        else:
            actor_info['used_capacity'] = 0.0
        new_actor = actor.apply_gradient_two(grads=grads_actor, has_aux=False)

    def temperature_loss_fn(temp_params: Params):
        temperature = temp.apply_fn({'params': temp_params})
        temp_loss = temperature * (actor_info['entropy'] - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}
    
    grads_temp, alpha_info = jax.grad(temperature_loss_fn, has_aux=True)(temp.params)
    # recording grads norm
    alpha_info['g_norm_temp'] = global_norm(grads_temp)
    new_temp = temp.apply_gradient(grads=grads_temp, has_aux=False)
    
    rng, key = jax.random.split(rng)
    dist, _ = actor(batch.next_observations, jnp.array([task_id]))
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
            'q2': q2.mean()}
    
    grads_critic, critic_info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    # recording grads norm
    critic_info['g_norm_critic'] = global_norm(grads_critic)
    new_critic = critic.apply_gradient(grads=grads_critic, has_aux=False)
    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info}


class SPCLearner(SACLearner):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        task_n: int,
        max_step: int = 1000000,
        finetune_steps: int = 10000,
        dict_configs: dict = {},
        optim_configs: dict = {},
        actor_configs: dict = {},
        critic_configs: dict = {},
        tau: float = 0.005,
        discount: float = 0.99,
        target_update_period: int = 1,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0):
        super(SPCLearner, self).__init__(seed, observations, actions, optim_configs,
              actor_configs, critic_configs, discount, tau, target_update_period,
              target_entropy, init_temperature)

        action_dim = actions.shape[-1]

        self.rng, actor_key = jax.random.split(self.rng, 2)
        actor_configs['task_num'] = task_n
        actor_configs['action_dim'] = action_dim
        actor_def = policies.HatTanhPolicy(**actor_configs)
        actor = AlterTrainableModel.create(
            model_def=actor_def,
            inputs=[actor_key, observations, jnp.array([0])],
            params_list_one=['embeds'],
            params_list_two=['backbones', 'mean', 'log'],
            optimizer_one=set_optimizer(**optim_configs),
            optimizer_two=set_optimizer(**optim_configs)
        )

        # get grads masking func
        def get_grad_masks(model, masks):
            g_masks = model.get_grad_masks(masks)
            return g_masks
        get_grad_masks_jit = jax.jit(nn.apply(get_grad_masks, actor_def))

        # preset dict learner for each layer:
        self.dict4layers = {}
        for id_layer, hidn in enumerate(actor_configs['hidden_dims']):
            dict_learner = OnlineDictLearnerV2(
                n_features=384,
                n_components=hidn,
                seed=seed + id_layer + 1,
                positive_code=True,
                verbose=True,
                **dict_configs)
            self.dict4layers[f'embeds_bb_{id_layer}'] = dict_learner

        # self.masks = {}
        # self.target_masks = {}

        self.actor = actor
        self.mask_cum = None
        self.mask_prm = None
        self.finetune = False
        self.fir_task = True
        self.max_step = max_step
        self.actor_cfgs = actor_configs
        self.finetune_steps = finetune_steps
        self.get_grad_masks = get_grad_masks_jit
        self.task_embeddings = []
        self.task_encoder = SentenceTransformer('all-MiniLM-L12-v2')

        self.invalid_filter = jax.jit(
            lambda x:
            jax.tree_map(
                lambda y: jnp.where(jnp.logical_or(jnp.isnan(y), jnp.isinf(y)), 0, y),
                x
            )
        )

    def start_task(self, task_id: int, description: str):
        task_e = self.task_encoder.encode(description)[np.newaxis]
        self.task_embeddings.append(task_e)

        # set init gates for each layer of actor
        actor_params = unfreeze(self.actor.params)
        for k in self.actor.params.keys():
            if k.startswith('embeds'):
                gates = self.dict4layers[k].get_gates(task_e)
                gates = jnp.asarray(gates.flatten())
                # self.masks[k] = gates.flatten()
                # if self.fir_task:
                #     self.target_masks[k] = gates.flatten()
                # Replace the i-th coefficients
                actor_params[k]['embedding'] = actor_params[k]['embedding'].at[task_id].set(gates.flatten())
        new_actor = self.actor.replace(params=freeze(actor_params))
        self.actor = new_actor

    # def ema_update_masks(self, task_id: int):
    #     actor_params = unfreeze(self.actor.params)
    #     for k in actor_params:
    #         if k.startswith('embeds'):
    #             scores = (1.0 - self.tau) * self.target_masks[k] + self.tau * self.masks[k]
    #             actor_params[k]['embedding'] = actor_params[k]['embedding'].at[task_id].set(scores)
    #             self.target_masks[k] = scores
    #     new_actor = self.actor.replace(params=freeze(actor_params))
    #     self.actor = new_actor

    def sample_actions(self,
                       observations: np.ndarray,
                       task_id: int,
                       temperature: float = 1.0) -> jnp.ndarray:

        rng, actions, _ = _sample_actions(self.rng, self.actor.apply_fn,
                                          self.actor.params, observations, 
                                          jnp.array([task_id]), temperature)
        self.rng = rng

        actions = self.invalid_filter(actions)
        return actions

    def update(self, task_id: int, batch: Batch) -> InfoDict:
        if self.step == self.max_step - self.finetune_steps:
            self.finetune = True

        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_spc_jit(
            rng=self.rng, task_id=task_id, pms_mask=self.mask_prm, cum_mask=self.mask_cum,
            actor=self.actor, critic=self.critic, target_critic=self.target_critic, 
            temp=self.temp, batch=batch, discount=self.discount, tau=self.tau, target_entropy=self.target_entropy, 
            finetune=self.finetune, first_task=self.fir_task)

        self.step += 1
        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp     

        return info

    def end_task(self, task_id: int):
        self.step = 0

        self.rng, _, dicts = _sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, self.dummy_o, 
                                             jnp.array([task_id]))
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
            lambda x: jnp.where(x < 1.0, 0, 1.0), grad_masks)
        self.mask_prm = freeze(grad_masks)

        # reset params for critic ,target_critic and temperature
        self.rng, key_critic, key_temp, key_actor = jax.random.split(self.rng, 4)

        self.critic = utils_fn.reset_model(
            self.critic, 
            critic_net.DoubleCritic,
            self.critic_cfgs,
            [key_critic, self.dummy_o, self.dummy_a])

        self.target_critic = self.target_critic.update_params(self.critic.params)

        self.temp = utils_fn.reset_model(
            self.temp,
            temperature.Temperature,
            {'init_log_temp': self.init_temp},
            [key_temp])

        # reset unused params for actor
        self.actor = utils_fn.reset_part_params(
            self.actor.params,
            self.mask_prm,
            self.actor,
            policies.HatTanhPolicy,
            self.actor_cfgs,
            [key_actor, self.dummy_o, jnp.array([0])],
            independent=False)

        # reset log_std_layer params
        new_params_actor = utils_fn.reset_logstd_layer(
            key_actor,
            self.actor.params,
            self.actor_cfgs['final_fc_init_scale'],
            self.actor_cfgs['state_dependent_std'])
        self.actor.update_params(new_params_actor)

        # update dictionary learners
        for k in self.actor.params.keys():
            if k.startswith('embeds'):
                optimal_gates = self.actor.params[k]['embedding'][task_id]
                optimal_gates = np.array([optimal_gates.flatten()])
                task_e = self.task_embeddings[task_id]
                # online update dictionary via CD
                self.dict4layers[k].update_dict(optimal_gates, task_e)

        # reset optimizers
        self.actor = self.actor.reset_optimizer()
        self.critic = self.critic.reset_optimizer()
        self.temp = self.temp.reset_optimizer()

        self.finetune = False  # close finetune for the next task
        self.fir_task = False  # trigger for the next task
