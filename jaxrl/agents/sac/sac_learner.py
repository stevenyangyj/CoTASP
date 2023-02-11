"""Implementations of algorithms for continuous control."""

from copy import deepcopy
import functools
from typing import Optional, Tuple, Callable, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from flax import linen as nn
from flax.core import freeze, unfreeze, FrozenDict
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import sparse_encode

import jaxrl.networks.common as utils_fn
from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.actor import update as update_actor
from jaxrl.agents.sac.critic import target_update
from jaxrl.agents.sac.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, TrainState, PRNGKey, Params, \
    MPNTrainState
from jaxrl.dict_learning.task_dict import OnlineDictLearnerV2


@jax.jit
def _update_sac_jit(
    rng: PRNGKey, actor: TrainState, critic: TrainState, target_critic: TrainState,
    temp: TrainState, batch: Batch, discount: float, tau: float,
    target_entropy: float
    ) -> Tuple[PRNGKey, TrainState, TrainState, TrainState, TrainState, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount)

    new_target_critic = target_update(new_critic, target_critic, tau)

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
        _, actor_params = actor_def.init(actor_key, observations).pop('params')
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=utils_fn.set_optimizer(**optim_configs)
        )

        critic_def = critic_net.DoubleCritic(**critic_configs)
        _, critic_params = critic_def.init(
            critic_key, observations, actions
        ).pop('params')
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=utils_fn.set_optimizer(**optim_configs)
        )

        target_critic = deepcopy(critic)

        opt_kwargs_temp = deepcopy(optim_configs)
        opt_kwargs_temp['max_norm'] = -1.0
        opt_kwargs_temp['clip_method'] = None
        temp_def = temperature.Temperature(init_temperature)
        _, temp_params = temp_def.init(temp_key).pop('params')
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=utils_fn.set_optimizer(**opt_kwargs_temp)
        )

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

        self.invalid_filter = jax.jit(
            lambda x:
            tree_map(
                lambda y: jnp.where(jnp.logical_or(jnp.isnan(y), jnp.isinf(y)), 0, y),
                x
            )
        )

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = self.invalid_filter(actions)
        return actions

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1

        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_sac_jit(
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info

    def end_task(self, save_actor_dir: str):

        # reset params for critic ,target_critic and temperature
        self.rng, key_critic, key_temp = jax.random.split(self.rng, 3)

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

        # reset optimizers
        self.actor = self.actor.reset_optimizer()
        self.critic = self.critic.reset_optimizer()
        self.temp = self.temp.reset_optimizer()

        if save_actor_dir is not None:
            # save actor params
            self.actor.save(save_actor_dir)


@jax.jit
def _sample_actions(
        rng: PRNGKey,
        actor: MPNTrainState,
        observations: np.ndarray,
        task_i: jnp.ndarray,
        temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray, dict]:
    rng, key = jax.random.split(rng)
    dist, dicts = actor(
        observations,
        task_i,
        temperature
    )

    return rng, dist.sample(seed=key), dicts


def _update_alpha(
    rng: PRNGKey, task_id: int, param_mask: FrozenDict[str, Any],
    actor: MPNTrainState, critic: TrainState, temp: TrainState, 
    batch: Batch) -> Tuple[PRNGKey, MPNTrainState, InfoDict]:

    rng, key = jax.random.split(rng)
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, dicts = actor.apply_fn(
            {'params': actor_params}, batch.observations, jnp.array([task_id])
        )
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()

        # if finetune:
        #     # masking regularization
        #     reg = 0
        #     count = 0
        #     if first_task:
        #         for m_k in dicts['masks']:
        #             reg += dicts['masks'][m_k]['embedding'].sum()
        #             count += dicts['masks'][m_k]['embedding'].size
        #     else:
        #         for m_k in dicts['masks']:
        #             aux = 1 - cumul_mask[m_k]['embedding']
        #             reg += (dicts['masks'][m_k]['embedding'] * aux).sum()
        #             count += aux.sum()
        #     regularizer = reg / count * jax.lax.stop_gradient(0.5 * jnp.sqrt(jnp.abs(q).mean()))

        #     actor_loss += regularizer

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
            _info[k+'_rate_act'] = jnp.mean(dicts['masks'][k])

        return actor_loss, _info
    
    # grads of actor
    grads_actor, actor_info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    # recording info
    actor_info['g_norm_actor'] = utils_fn.global_norm(grads_actor)
    actor_info['used_capacity'] = 1.0 - utils_fn.rate_activity(param_mask)

    # only update coefficients (alpha)
    new_actor = actor.apply_grads_alpha(grads=grads_actor)

    return rng, new_actor, actor_info

def _update_theta(
    rng: PRNGKey, task_id: int, param_mask: FrozenDict[str, Any], 
    actor: MPNTrainState, critic: TrainState, temp: TrainState, 
    batch: Batch) -> Tuple[PRNGKey, MPNTrainState, InfoDict]:

    rng, key = jax.random.split(rng)
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, dicts = actor.apply_fn(
            {'params': actor_params}, batch.observations, jnp.array([task_id])
        )
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()

        _info = {
            'hac_sac_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'means': dicts['means'].mean()
        }
        for k in dicts['masks']:
            _info[k+'_rate_act'] = jnp.mean(dicts['masks'][k])

        return actor_loss, _info
    
    # grads of actor
    grads_actor, actor_info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    # recording info
    actor_info['g_norm_actor'] = utils_fn.global_norm(grads_actor)
    actor_info['used_capacity'] = 1.0 - utils_fn.rate_activity(param_mask)

    # Maksing gradients according to cumulative binary masks
    unfrozen_grads = unfreeze(grads_actor)
    for path, value in param_mask.items():
        cursor = unfrozen_grads
        for key in path[:-1]:
            if key in cursor:
                cursor = cursor[key]
        cursor[path[-1]] *= value
    
    new_actor = actor.apply_grads_theta(grads=freeze(unfrozen_grads))

    return rng, new_actor, actor_info

def _update_temp(
    temp: TrainState, actor_entropy: float, target_entropy: float
    ) -> Tuple[TrainState, InfoDict]:

    def temperature_loss_fn(temp_params: Params):
        temperature = temp.apply_fn({'params': temp_params})
        temp_loss = temperature * (actor_entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}
    
    grads_temp, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(temp.params)
    # recording info
    temp_info['g_norm_temp'] = utils_fn.global_norm(grads_temp)

    new_temp = temp.apply_gradients(grads=grads_temp)

    return new_temp, temp_info

def _update_critic(
    rng: PRNGKey, task_id: int, actor: MPNTrainState, critic: TrainState, 
    target_critic: TrainState, temp: TrainState, batch: Batch, discount: float, 
    tau: float) -> Tuple[PRNGKey, TrainState, TrainState, InfoDict]:
    
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
    # recording info
    critic_info['g_norm_critic'] = utils_fn.global_norm(grads_critic)

    new_critic = critic.apply_gradients(grads=grads_critic)
    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_critic, new_target_critic, critic_info


@jax.jit
def _update_cotasp_jit(rng: PRNGKey, task_id: int, tau: float, discount: float, 
    target_entropy: float, optimize_alpha: bool, param_mask: FrozenDict[str, Any], 
    actor: MPNTrainState, critic: TrainState, target_critic: TrainState, 
    temp: TrainState, batch: Batch
    ) -> Tuple[PRNGKey, MPNTrainState, TrainState, TrainState, TrainState, InfoDict]:
    # optimizing critics
    new_rng, new_critic, new_target_critic, critic_info = _update_critic(
        rng, task_id, actor, critic, target_critic, temp, batch, discount, tau
    )

    # optimizing either alpha or theta
    new_rng, new_actor, actor_info = jax.lax.cond(
        optimize_alpha,
        _update_alpha,
        _update_theta,
        new_rng, task_id, param_mask, actor, new_critic, temp, batch
    )

    # updating temperature coefficient
    new_temp, temp_info = _update_temp(
        temp, actor_info['entropy'], target_entropy
    )

    return new_rng, new_actor, new_temp, new_critic, new_target_critic, {
        **actor_info,
        **temp_info,
        **critic_info
    }


class CoTASPLearner(SACLearner):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        task_num: int,
        load_policy_dir: Optional[str] = None,
        load_dict_dir: Optional[str] = None,
        update_dict = True,
        update_coef = True,
        dict_configs: dict = {},
        optim_configs: dict = {},
        actor_configs: dict = {},
        critic_configs: dict = {},
        tau: float = 0.005,
        discount: float = 0.99,
        target_update_period: int = 1,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0):
        super(CoTASPLearner, self).__init__(seed, observations, actions, optim_configs,
              actor_configs, critic_configs, discount, tau, target_update_period,
              target_entropy, init_temperature)

        action_dim = actions.shape[-1]

        self.rng, actor_key = jax.random.split(self.rng, 2)
        actor_configs['task_num'] = task_num
        actor_configs['action_dim'] = action_dim
        actor_def = policies.MetaPolicy(**actor_configs)
        _, actor_params = actor_def.init(actor_key, observations, jnp.array([0])).pop('params')
        actor = MPNTrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=utils_fn.set_optimizer(**optim_configs)
        )

        if load_policy_dir is not None:
            actor = actor.load(load_policy_dir)

        # get grads masking func
        def get_grad_masks(model, masks):
            g_masks = model.get_grad_masks(masks)
            return g_masks
        get_grad_masks_jit = jax.jit(nn.apply(get_grad_masks, actor_def))

        # preset dict learner for each layer:
        self.dict4layers = {}
        for id_layer, hidn in enumerate(actor_configs['hidden_dims']):
            dict_learner = OnlineDictLearnerV2(
                384,
                hidn,
                seed+id_layer+1,
                None, # whether using svd dictionary initialization
                **dict_configs)
            self.dict4layers[f'embeds_bb_{id_layer}'] = dict_learner
        
        if load_dict_dir is not None:
            for k in self.dict4layers.keys():
                self.dict4layers[k].load(f'{load_dict_dir}/{k}.pkl')

        # initialize param_masks
        self.rng, _, dicts = _sample_actions(
            self.rng, actor, self.dummy_o, jnp.array([0])
        )
        self.cumul_masks = tree_map(lambda x: jnp.zeros_like(x), dicts['masks'])
        self.param_masks = freeze(
            get_grad_masks_jit({'params': actor.params}, self.cumul_masks)
        )

        # initialize other things
        self.actor = actor
        self.update_dict = update_dict
        self.update_coef = update_coef
        self.actor_cfgs = actor_configs
        self.get_grad_masks = get_grad_masks_jit
        self.task_embeddings = []
        self.task_encoder = SentenceTransformer('all-MiniLM-L12-v2')

    def start_task(self, task_id: int, description: str):
        task_e = self.task_encoder.encode(description)[np.newaxis, :]
        self.task_embeddings.append(task_e)

        # set initial alpha for each layer of MPN
        actor_params = unfreeze(self.actor.params)
        for k in self.actor.params.keys():
            if k.startswith('embeds'):
                alpha_l = self.dict4layers[k].get_alpha(task_e)
                alpha_l = jnp.asarray(alpha_l.flatten())
                # Replace the i-th row
                actor_params[k]['embedding'] = actor_params[k]['embedding'].at[task_id].set(alpha_l)
        self.actor = self.actor.update_params(freeze(actor_params))

    def sample_actions(self,
                       observations: np.ndarray,
                       task_id: int,
                       temperature: float = 1.0) -> jnp.ndarray:

        rng, actions, _ = _sample_actions(self.rng, self.actor, observations, 
                                          jnp.array([task_id]), temperature)
        self.rng = rng
        actions = self.invalid_filter(actions)

        return actions

    def update(self, task_id: int, batch: Batch, optimize_alpha: bool=False) -> InfoDict:

        if not self.update_coef:
            optimize_alpha = False

        new_rng, new_actor, new_temp, new_critic, new_target_critic, info = _update_cotasp_jit(
            self.rng, task_id, self.tau, self.discount, self.target_entropy, optimize_alpha, 
            self.param_masks, self.actor, self.critic, self.target_critic, self.temp, batch
        )

        self.step += 1
        self.rng = new_rng
        self.actor = new_actor
        self.temp = new_temp  
        self.critic = new_critic
        self.target_critic = new_target_critic   

        return info

    def reset_agent(self):
        # re-initialize params of critic ,target_critic and temperature
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

        # re-initialize unused params of meta-policy network
        new_rng, new_params = utils_fn.reset_free_params(
            self.actor.params,
            self.param_masks,
            policies.MetaPolicy,
            self.actor_cfgs,
            [key_actor, self.dummy_o, jnp.array([0])],
            adaptive_init=True
        )
        # re-initialize log_std_layer's params
        self.rng, new_params = utils_fn.reset_logstd_layer(
            new_rng,
            new_params,
            self.actor_cfgs['final_fc_init_scale'],
            self.actor_cfgs['state_dependent_std']
        )
        self.actor = self.actor.update_params(new_params)

        # reset optimizers
        self.actor = self.actor.reset_optimizer()
        self.critic = self.critic.reset_optimizer()
        self.temp = self.temp.reset_optimizer()

    def end_task(self, task_id: int, save_actor_dir: str, save_dict_dir: str):
        self.step = 0

        self.rng, _, dicts = _sample_actions(
            self.rng, self.actor, self.dummy_o, jnp.array([task_id])
        )
        current_masks = dicts['masks']

        # cumulative attention from all the previous tasks
        self.cumul_masks = tree_map(
            lambda a, b: jnp.maximum(a, b), self.cumul_masks, current_masks
        )

        # To condition the training of task t + 1, we compute 
        # the gradient masks according to cumulative binary masks
        grad_masks = self.get_grad_masks(
            {'params': self.actor.params}, self.cumul_masks
        )
        self.param_masks = freeze(grad_masks)

        # update dictionary learners
        dict_stats = {}
        if self.update_dict:
            for k in self.actor.params.keys():
                if k.startswith('embeds'):
                    optimal_alpha_l = self.actor.params[k]['embedding'][task_id]
                    optimal_alpha_l = np.array([optimal_alpha_l.flatten()])
                    task_e = self.task_embeddings[task_id]
                    # online update dictionary via CD
                    self.dict4layers[k].update_dict(optimal_alpha_l, task_e)
                    dict_stats[k] = {
                        'sim_mat': self.dict4layers[k]._compute_overlapping(),
                        'change_of_d': np.array(self.dict4layers[k].change_of_dict)
                    }
        else:
            for k in self.actor.params.keys():
                if k.startswith('embeds'):
                    dict_stats[k] = {
                        'sim_mat': self.dict4layers[k]._compute_overlapping(),
                        'change_of_d': 0
                    }

        self.reset_agent()

        if save_actor_dir is not None and save_dict_dir is not None:
            # save actor params
            self.actor.save(save_actor_dir)
            # save dicts
            for k in self.dict4layers.keys():
                self.dict4layers[k].save(f'{save_dict_dir}/{k}.pkl')

        return dict_stats

    def freeze_task_params(self, task_id):
        self.rng, _, dicts = _sample_actions(self.rng, self.actor, 
                                             self.dummy_o, jnp.array([task_id]))
        current_masks = dicts['masks']
        # cumulative attention from all the previous tasks
        self.cumul_masks = tree_map(
            lambda a, b: jnp.maximum(a, b), self.cumul_masks, current_masks
        )

        # To condition the training of task t + 1, we compute 
        # the gradient masks according to cumulative binary masks
        grad_masks = self.get_grad_masks(
            {'params': self.actor.params}, self.cumul_masks)
        self.param_masks = freeze(grad_masks)


class TaDeLL(SACLearner):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        load_policy_dir: Optional[str] = None,
        load_dict_dir: Optional[str] = None,
        dict_configs: dict = {},
        optim_configs: dict = {},
        actor_configs: dict = {},
        critic_configs: dict = {},
        tau: float = 0.005,
        discount: float = 0.99,
        target_update_period: int = 1,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0):
        super(TaDeLL, self).__init__(seed, observations, actions, optim_configs,
              actor_configs, critic_configs, discount, tau, target_update_period,
              target_entropy, init_temperature)

        flat_actor_params, pack_fn = ravel_pytree(unfreeze(self.actor.params))
        feature_dim = len(flat_actor_params) + 384
        self.dict_learner = OnlineDictLearnerV2(
            feature_dim,
            20,
            seed,
            None, # whether using svd dictionary initialization
            **dict_configs
        )

        if load_policy_dir is not None:
            self.actor = self.actor.load(load_policy_dir)
        if load_dict_dir is not None:
            self.dict_learner.load(f'{load_dict_dir}.pkl')

        self.pack_jit = jax.jit(pack_fn)
        self.dict_cfgs = dict_configs
        self.num_params = len(flat_actor_params)
        self.eval_actor = deepcopy(self.actor)
        self.task_params = []
        self.task_encoder = SentenceTransformer('all-MiniLM-L12-v2')

    def start_task(self, description: str):
        self.task_emb = self.task_encoder.encode(description)[np.newaxis]

        # set actor parameters via sparse coding of task embedding
        task_code = sparse_encode(
            self.task_emb,
            self.dict_learner.D[:, self.num_params:],
            algorithm=self.dict_cfgs['method'], 
            alpha=self.dict_cfgs['alpha'],
            check_input=False,
            positive=self.dict_cfgs['positive_code'],
            max_iter=10000)
        
        recon_params = np.dot(
            task_code, self.dict_learner.D[:, :self.num_params]
        )

        new_actor_params = self.pack_jit(recon_params.flatten())
        self.actor = self.actor.update_params(freeze(new_actor_params))

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0,
                       eval_mode: bool = False) -> jnp.ndarray:

        if eval_mode:
            rng, actions = policies.sample_actions(self.rng, self.eval_actor.apply_fn,
                                                   self.eval_actor.params, observations,
                                                   temperature)
        else:
            rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                                   self.actor.params, observations,
                                                   temperature)
        self.rng = rng

        actions = self.invalid_filter(actions)
        return actions

    def select_actor(self, task_id):
        if task_id < len(self.task_params):
            self.eval_actor = self.eval_actor.update_params(self.task_params[task_id])
        else:
            self.eval_actor = self.eval_actor.update_params(self.actor.params)

    def end_task(self, save_actor_dir: str):
        # update dictionary
        self.task_params.append(deepcopy(self.actor.params))
        flat_actor_params, _ = ravel_pytree(unfreeze(self.actor.params))
        sample = np.hstack([flat_actor_params[np.newaxis], self.task_emb])
        s_code = self.dict_learner.get_alpha(sample)
        self.dict_learner.update_dict(s_code, sample)

        # reset params for critic ,target_critic and temperature
        self.rng, key_critic, key_temp = jax.random.split(self.rng, 3)

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

        # reset optimizers
        self.actor = self.actor.reset_optimizer()
        self.critic = self.critic.reset_optimizer()
        self.temp = self.temp.reset_optimizer()

        if save_actor_dir is not None:
            # save actor params
            self.actor.save(save_actor_dir)

