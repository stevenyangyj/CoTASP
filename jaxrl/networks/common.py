import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jax.tree_util import tree_map
from flax.core import freeze, unfreeze, FrozenDict
from flax import traverse_util
from flax import struct
from flax import core


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, 'fan_out', 'normal')


def activation_fn(name: str = 'lrelu'):
    assert name in ['relu', 'tanh', 'leaky_relu', 'swish', 'elu',
                    'gelu', 'selu', 'celu', 'hard_tanh']
    if name == 'relu':
        act_fn = nn.relu
    elif name == 'tanh':
        act_fn = nn.tanh
    elif name == 'leaky_relu':
        act_fn = jax.nn.leaky_relu
    elif name == 'swish':
        act_fn = jax.nn.swish
    elif name == 'elu':
        act_fn = nn.elu
    elif name == 'gelu':
        act_fn = jax.nn.gelu
    elif name == 'selu':
        act_fn = jax.nn.selu
    elif name == 'celu':
        act_fn = jax.nn.celu
    elif name == 'hard_tanh':
        act_fn = jax.nn.hard_tanh
    return act_fn


def flattened_traversal(fn):
    """Returns function that is called with `(path, param)` instead of pytree."""
    def mask(tree):
        flat = flax.traverse_util.flatten_dict(tree)
        return flax.traverse_util.unflatten_dict(
            {k: fn(k, v) for k, v in flat.items()})
    return mask


def create_mask(params, trainable_names):
    def _map(params, mask, names):
        for k in params:
            if k.split('_')[0] in names:
                mask[k] = 'train'
            else:
                mask[k] = 'fix'

    mask = {}
    _map(params, mask, trainable_names)
    return freeze(mask)


def filter_theta(path, _):
    for i in range(10):
        if f'backbones_{i}' in path:
            return 'frozen'
    if 'mean_layer' in path:
        return 'frozen'
    elif 'log_std_layer' in path:
        return 'frozen'
    else:
        return 'trainable'


def filter_alpha(path, _):
    for i in range(10):
        if f'embeds_bb_{i}' in path:
            return 'frozen'
    return 'trainable'


def abs_sq(x: jnp.array) -> jnp.array:
    """Returns the squared norm of a (maybe complex) array.
    For real `x`, JAX generates the same HLO from this, `jnp.square(x)`, `x * x`,
    or `x**2`.
    Args:
        x: a (maybe complex) array.
    Returns:
        The squared norm of `x`.
    """
    if not isinstance(x, (np.ndarray, jnp.ndarray)):
        raise ValueError(f"`abs_sq` accepts only NDarrays, got: {x}.")
    return (x.conj() * x).real


def global_norm(updates: Params) -> Params:
    """Compute the global norm across a nested structure of tensors."""
    return jnp.sqrt(sum(
        jnp.sum(abs_sq(x)) for x in jax.tree_util.tree_leaves(updates)))


def tree_l1_mean(updates: Params) -> jnp.array:
    count = 0
    sums = 0
    for x in jax.tree_util.tree_leaves(updates):
        count += x.size
        sums += jax.lax.abs(x).sum()
    
    return sums / count


def rate_activity(updates: Params) -> jnp.array:
    sums = 0
    count = 0
    for x in jax.tree_util.tree_leaves(updates):
        count += x.size
        sums += x.sum()
    
    return sums / count


def replace_embeds(pi_params: Params, codes: FrozenDict, components: FrozenDict, index: int) -> Params:
    actor_params = unfreeze(pi_params)
    for k in components.keys():
        embeds = jnp.dot(codes[k]['codes'], components[k]['components'])
        actor_params[k]['embedding'] = actor_params[k]['embedding'].at[index].set(embeds.flatten())
    return freeze(actor_params)


def reset_model(main_cls, model_cls, configs: dict, inputs: list):
    model = model_cls(**configs)
    _, new_params = model.init(*inputs).pop('params')
    return main_cls.update_params(new_params)


def reset_logstd_layer(
    rng: PRNGKey, pi_params: Params, 
    final_fc_init_scale: float, state_dependent_std: bool):

    params_actor = unfreeze(pi_params)
    if state_dependent_std:
        kernel = params_actor['log_std_layer']['kernel']
        bias = params_actor['log_std_layer']['bias']

        rng, key = jax.random.split(rng)
        # init_kernel = default_init(scale)(key, kernel.shape, jnp.float32)
        init_kernel = default_init(final_fc_init_scale)(key, kernel.shape)
        init_bias = nn.initializers.zeros_init()(key, bias.shape)

        params_actor['log_std_layer']['kernel'] = init_kernel
        params_actor['log_std_layer']['bias'] = init_bias
    else:
        params_actor['log_std_layer'] = jnp.zeros_like(params_actor['log_std_layer'])
    return rng, freeze(params_actor)


def reset_free_params(
    params: Params, 
    param_masks: Params,
    model_cls, 
    configs: dict, 
    inputs: list,
    adaptive_init: bool=True):
    if not adaptive_init:
        model = model_cls(**configs)
        _, init_params = model.init(*inputs).pop('params')
    else:
        rng, _ = jax.random.split(inputs[0])
        init_params = {}
        for path, value in param_masks.items():
            cursor = init_params
            for key in path[:-1]:
                if key not in cursor:
                    cursor[key] = {}
                cursor = cursor[key]
            rng, key = jax.random.split(rng)
            if path[-1] == 'kernel':
                if path[0] == 'mean_layer':
                    rescale = jnp.mean(value) * configs['final_fc_init_scale']
                else:
                    rescale = jnp.mean(value)
                cursor[path[-1]] = default_init(rescale)(key, value.shape)
            elif path[-1] == 'bias':
                cursor[path[-1]] = nn.initializers.zeros_init()(key, value.shape)
            else:
                raise NotImplementedError

    new_params = unfreeze(params)
    for k in init_params:
        for sub_k in init_params[k]:
            new_params[k][sub_k] = new_params[k][sub_k] * (1.0 - param_masks[(k, sub_k)]) \
                + init_params[k][sub_k] * param_masks[(k, sub_k)]

    return rng, freeze(new_params)


def set_optimizer(
    lr: float, 
    max_norm: float, 
    optim_algo: str='adam', 
    clip_method: str='global_clip',
    decay_coeff: Optional[float]=None) -> optax.GradientTransformation:

    if optim_algo == 'adam':
        optimizer = optax.adam(learning_rate=lr)
    elif optim_algo == 'adamw':
        optimizer = optax.adamw(learning_rate=lr, weight_decay=decay_coeff)
    elif optim_algo == 'lamb':
        optimizer = optax.lamb(learning_rate=lr)
    elif optim_algo == 'sgd':
        optimizer = optax.sgd(learning_rate=lr)
    elif optim_algo == 'radam':
        optimizer = optax.radam(learning_rate=lr)
    elif optim_algo == 'adabelief':
        optimizer = optax.adabelief(learning_rate=lr)
    elif optim_algo == 'amsgrad':
        optimizer = optax.amsgrad(learning_rate=lr)
    else:
        raise NotImplementedError

    if clip_method == 'global_clip':
        minimizer = optax.chain(
            optax.clip_by_global_norm(max_norm),
            optimizer)
    elif clip_method == 'clip':
        minimizer = optax.chain(
            optax.clip(max_norm),
            optimizer)
    elif clip_method == 'adaptive_clip':
        minimizer = optax.chain(
            optax.adaptive_grad_clip(max_norm),
            optimizer)
    elif max_norm == -1.0:
        minimizer = optimizer
    else:
        raise NotImplementedError

    return minimizer


def linear_warm_start(
    start_val: float, 
    end_val: float, 
    warm_steps: int, 
    linear_steps: int) -> optax.Schedule:
    list_schedules = [
        optax.linear_schedule(
            init_value=end_val,
            end_value=start_val,
            transition_steps=warm_steps,
            transition_begin=0),
        optax.linear_schedule(
            init_value=start_val,
            end_value=end_val,
            transition_steps=linear_steps,
            transition_begin=0)
    ]
    return optax.join_schedules(list_schedules, [warm_steps])


def linear_ascent(
    start_val: float, 
    end_val: float, 
    linear_steps: int) -> optax.Schedule:
    return optax.linear_schedule(
        init_value=start_val,
        end_value=end_val,
        transition_steps=linear_steps,
        transition_begin=0)


class RMSNorm(nn.Module):
    axis: int
    eps: float = 1e-5
    create_scale: bool = False
    scale_init: Optional[jax.nn.initializers.Initializer] = None

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if self.create_scale:
            scale = self.param('scale', self.scale_init, inputs.shape)
            scale = jnp.broadcast_to(scale, inputs.shape)
        else:
            scale = 1.
        
        mean_squared = jnp.mean(jnp.square(inputs), axis=self.axis, keepdims=True)
        mean_squared = jnp.broadcast_to(mean_squared, inputs.shape)
        
        return inputs * scale * jax.lax.rsqrt(mean_squared + self.eps)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            # whether using layer normalization
            if i == 0 and self.use_layer_norm:
                x = nn.LayerNorm()(x)
                x = nn.tanh(x)
            else:
                if i + 1 < len(self.hidden_dims) or self.activate_final:
                    x = self.activations(x)
        return x


class TrainState(struct.PyTreeNode):
    """Simple train state for the common case with a single Optax optimizer.

    Synopsis::

        state = TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=tx)
        grad_fn = jax.grad(make_loss_fn(state.apply_fn))
        for batch in data:
        grads = grad_fn(state.params, batch)
        state = state.apply_gradients(grads=grads)

    Note that you can easily extend this dataclass by subclassing it for storing
    additional data (e.g. additional variable collections).

    For more exotic usecases (e.g. multiple optimizers) it's probably best to
    fork the class and modify it.

    Args:
    step: Counter starts at 0 and is incremented by every call to
        `.apply_gradients()`.
    apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
        convenience to have a shorter params list for the `train_step()` function
        in your training loop.
    params: The parameters to be updated by `tx` and used by `apply_fn`.
    tx: An Optax gradient transformation.
    opt_state: The state for `tx`.
    """
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState

    def __call__(self, *args, **kwargs):
        return self.apply_fn({'params': self.params}, *args, **kwargs)

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
        grads: Gradients that have the same pytree structure as `.params`.
        **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
        An updated instance of `self` with `step` incremented by one, `params`
        and `opt_state` updated by applying `grads`, and additional attributes
        replaced as specified by `kwargs`.
        """
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def update_params(self, new_params: core.FrozenDict[str, Any]) -> 'TrainState':
        return self.replace(params=new_params)

    def reset_optimizer(self) -> 'TrainState': 
        # contain the count argument
        init_opt_state = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x), self.opt_state
        )
        return self.replace(opt_state=init_opt_state)

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'TrainState':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
            params = jax.tree_util.tree_map(jnp.array, params)
        return self.replace(params=params)

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


class MPNTrainState(struct.PyTreeNode):
    # Simple train state allowing alternate optimization.

    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    tx_theta: optax.GradientTransformation = struct.field(pytree_node=False)
    tx_alpha: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state_theta: optax.OptState
    opt_state_alpha: optax.OptState

    def __call__(self, *args, **kwargs):
        return self.apply_fn({'params': self.params}, *args, **kwargs) 

    def apply_grads_theta(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx_theta.update(
            grads, self.opt_state_theta, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state_theta=new_opt_state,
            **kwargs,
        )

    def apply_grads_alpha(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx_alpha.update(
            grads, self.opt_state_alpha, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state_alpha=new_opt_state,
            **kwargs,
        )

    def update_params(self, new_params: core.FrozenDict[str, Any]) -> 'MPNTrainState':
        return self.replace(params=new_params)

    def reset_optimizer(self):
        # contain the count argument
        opt_state_theta = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x),
                                                 self.opt_state_theta)
        opt_state_alpha = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x),
                                                 self.opt_state_alpha)
        return self.replace(opt_state_theta=opt_state_theta,
                            opt_state_alpha=opt_state_alpha)

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'MPNTrainState':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
            params = jax.tree_util.tree_map(jnp.array, params)
        return self.replace(params=params)

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        # opt_state = tx.init(params)
        partition_optimizers = {'trainable': tx, 'frozen': optax.set_to_zero()}

        # theta optimizer
        param_theta = freeze(traverse_util.path_aware_map(filter_alpha, params))
        tx_theta = optax.multi_transform(partition_optimizers, param_theta)
        # alpha optimizer
        param_alpha = freeze(traverse_util.path_aware_map(filter_theta, params))
        tx_alpha = optax.multi_transform(partition_optimizers, param_alpha)

        # init optimizer
        opt_state_theta = tx_theta.init(params)
        opt_state_alpha = tx_alpha.init(params)

        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx_theta=tx_theta,
            tx_alpha=tx_alpha,
            opt_state_theta=opt_state_theta,
            opt_state_alpha=opt_state_alpha,
            **kwargs,
        )


# TODO: Replace with TrainState when it's ready
# https://github.com/google/flax/blob/master/docs/flip/1009-optimizer-api.md#train-state
# @flax.struct.dataclass
# class Model:
#     step: int
#     apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
#     params: Params
#     tx: Optional[optax.GradientTransformation] = flax.struct.field(
#         pytree_node=False)
#     opt_state: Optional[optax.OptState] = None

#     @classmethod
#     def create(cls,
#                model_def: nn.Module,
#                inputs: Sequence[jnp.ndarray],
#                tx: Optional[optax.GradientTransformation] = None
#                ) -> 'Model':
#         variables = model_def.init(*inputs)

#         _, params = variables.pop('params')

#         if tx is not None:
#             opt_state = tx.init(params)
#         else:
#             opt_state = None

#         return cls(step=1,
#                    apply_fn=model_def.apply,
#                    params=params,
#                    tx=tx,
#                    opt_state=opt_state)

#     def __call__(self, *args, **kwargs):
#         return self.apply_fn({'params': self.params}, *args, **kwargs)

#     def apply_gradient(
#             self,
#             loss_fn: Optional[Callable[[Params], Any]] = None,
#             grads: Optional[Any] = None,
#             has_aux: bool = True) -> Union[Tuple['Model', Any], 'Model']:

#         if grads is None:
#             grad_fn = jax.grad(loss_fn, has_aux=has_aux)
#             if has_aux:
#                 _grads, aux = grad_fn(self.params)
#             else:
#                 _grads = grad_fn(self.params)
#         else:
#             _grads = grads

#         updates, new_opt_state = self.tx.update(_grads, self.opt_state,
#                                                 self.params)
#         new_params = optax.apply_updates(self.params, updates)

#         new_model = self.replace(step=self.step + 1,
#                                  params=new_params,
#                                  opt_state=new_opt_state)
#         if has_aux:
#             return new_model, aux
#         else:
#             return new_model

#     def update_params(self, new_params: Params) -> 'Model':
#         return self.replace(params=new_params)
    
#     def reset_optimizer(self) -> 'Model': 
#         # contain the count argument
#         init_opt_state = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x),
#                                                 self.opt_state)
#         return self.replace(opt_state=init_opt_state)

#     def save(self, save_path: str):
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         with open(save_path, 'wb') as f:
#             f.write(flax.serialization.to_bytes(self.params))

#     def load(self, load_path: str) -> 'Model':
#         with open(load_path, 'rb') as f:
#             params = flax.serialization.from_bytes(self.params, f.read())
#             params = jax.tree_util.tree_map(jnp.array, params)
#         return self.replace(params=params)


# @flax.struct.dataclass
# class AlterTrainableModel:
#     step: int
#     apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
#     params: Params
#     optimizer_one: Optional[optax.GradientTransformation] = flax.struct.field(
#         pytree_node=False)
#     optimizer_two: Optional[optax.GradientTransformation] = flax.struct.field(
#         pytree_node=False)
#     opt_state_one: Optional[optax.OptState] = None
#     opt_state_two: Optional[optax.OptState] = None

#     @classmethod
#     def create(cls,
#                model_def: nn.Module,
#                inputs: Sequence[jnp.ndarray],
#                params_list_one: list = None,
#                params_list_two: list = None,
#                optimizer_one: Optional[optax.GradientTransformation] = None,
#                optimizer_two: Optional[optax.GradientTransformation] = None,
#                ) -> 'AlterTrainableModel':
#         variables = model_def.init(*inputs)

#         _, params = variables.pop('params')

#         tx_one = optax.multi_transform({'train': optimizer_one, 'fix': optax.set_to_zero()},
#             create_mask(params, params_list_one))
#         opt_state_one = tx_one.init(params)

#         # another set of trainable param
#         tx_two = optax.multi_transform({'train': optimizer_two, 'fix': optax.set_to_zero()},
#             create_mask(params, params_list_two))
#         opt_state_two = tx_two.init(params)

#         return cls(step=1,
#                    apply_fn=model_def.apply,
#                    params=params,
#                    optimizer_one=tx_one,
#                    optimizer_two=tx_two,
#                    opt_state_one=opt_state_one,
#                    opt_state_two=opt_state_two)

#     def __call__(self, *args, **kwargs):
#         return self.apply_fn({'params': self.params}, *args, **kwargs)

#     def apply_gradient_one(
#             self,
#             loss_fn: Optional[Callable[[Params], Any]] = None,
#             grads: Optional[Any] = None,
#             has_aux: bool = True
#             ) -> Union[Tuple['AlterTrainableModel', Any], 'AlterTrainableModel']:

#         if grads is None:
#             grad_fn = jax.grad(loss_fn, has_aux=has_aux)
#             if has_aux:
#                 _grads, aux = grad_fn(self.params)
#             else:
#                 _grads = grad_fn(self.params)
#         else:
#             _grads = grads

#         updates, new_opt_state = self.optimizer_one.update(_grads, 
#             self.opt_state_one, self.params)
#         new_params = optax.apply_updates(self.params, updates)

#         new_model = self.replace(step=self.step + 1,
#                                  params=new_params,
#                                  opt_state_one=new_opt_state)
#         if has_aux:
#             return new_model, aux
#         else:
#             return new_model

#     def apply_gradient_two(
#             self,
#             loss_fn: Optional[Callable[[Params], Any]] = None,
#             grads: Optional[Any] = None,
#             has_aux: bool = True
#             ) -> Union[Tuple['AlterTrainableModel', Any], 'AlterTrainableModel']:

#         if grads is None:
#             grad_fn = jax.grad(loss_fn, has_aux=has_aux)
#             if has_aux:
#                 _grads, aux = grad_fn(self.params)
#             else:
#                 _grads = grad_fn(self.params)
#         else:
#             _grads = grads

#         updates, new_opt_state = self.optimizer_two.update(_grads, 
#             self.opt_state_two, self.params)
#         new_params = optax.apply_updates(self.params, updates)

#         new_model = self.replace(step=self.step + 1,
#                                  params=new_params,
#                                  opt_state_two=new_opt_state)
#         if has_aux:
#             return new_model, aux
#         else:
#             return new_model

#     def update_params(self, new_params: Params) -> 'AlterTrainableModel':
#         return self.replace(params=new_params)
    
#     def reset_optimizer(self) -> 'AlterTrainableModel':
#         # contain the count argument
#         opt_state_one = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x),
#                                                 self.opt_state_one)
#         opt_state_two = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x),
#                                                 self.opt_state_two)
#         return self.replace(opt_state_one=opt_state_one,
#                             opt_state_two=opt_state_two)

#     def save(self, save_path: str):
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         with open(save_path, 'wb') as f:
#             f.write(flax.serialization.to_bytes(self.params))

#     def load(self, load_path: str) -> 'AlterTrainableModel':
#         with open(load_path, 'rb') as f:
#             params = flax.serialization.from_bytes(self.params, f.read())
#             params = jax.tree_util.tree_map(jnp.array, params)
#         return self.replace(params=params)


if __name__ == "__main__":

    a = {'1': jnp.array([0.0, 0.2, 0.8, 0.9, 1.0, 0.0, 0.5, 0.49])}
    print(jax.jit(rate_activity)(a))