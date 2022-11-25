import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import freeze, unfreeze, FrozenDict


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


def activation_fn(name: str = 'lrelu'):
    assert name in ['relu', 'tanh', 'leaky_relu', 'swish', 'elu',
                    'gelu', 'selu']
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


def zero_grads():
    # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
    def init_fn(_): 
        return ()
    def update_fn(updates, state, params=None):
        return jax.tree_util.tree_map(jnp.zeros_like, updates), ()
    return optax.GradientTransformation(init_fn, update_fn)


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]


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


def replace_embeds(pi_params: Params, codes: FrozenDict, components: FrozenDict, index: int) -> Params:
    actor_params = unfreeze(pi_params)
    for k in components.keys():
        embeds = jnp.dot(codes[k]['codes'], components[k]['components'])
        actor_params[k]['embedding'] = actor_params[k]['embedding'].at[index].set(embeds.flatten())
    return freeze(actor_params)


def reset_logstd_layer(key: PRNGKey, pi_params: Params, scale: float):
    params_actor = unfreeze(pi_params)
    kernel = params_actor['log_std_layer']['kernel']
    bias = params_actor['log_std_layer']['bias']

    init_kernel = default_init(scale)(key, kernel.shape, jnp.float32)
    init_bias = jnp.zeros_like(bias)

    params_actor['log_std_layer']['kernel'] = init_kernel
    params_actor['log_std_layer']['bias'] = init_bias
    return freeze(params_actor)


def set_optimizer(
    lr: float, 
    max_norm: float, 
    optim_algo: str='adam', 
    clip_method: str='global_clip',
    decay_coef: Optional[float]=None) -> optax.GradientTransformation:

    if optim_algo == 'adam':
        optimizer = optax.adam(learning_rate=lr)
    elif optim_algo == 'adamw':
        optimizer = optax.adamw(learning_rate=lr, weight_decay=decay_coef)
    elif optim_algo == 'sgd':
        optimizer = optax.sgd(learning_rate=lr)
    elif optim_algo == 'radam':
        optimizer = optax.radam(learning_rate=lr)
    elif optim_algo == 'adabelief':
        optimizer = optax.adabelief(learning_rate=lr)
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
    dropout_rate: Optional[float] = None
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            # whether using layer normalization
            if i == 0 and self.use_layer_norm:
                x = nn.LayerNorm()(x)
                x = nn.tanh(x)
            else:
                if i + 1 < len(self.hidden_dims) or self.activate_final:
                    x = self.activations(x)
                    if self.dropout_rate is not None:
                        x = nn.Dropout(rate=self.dropout_rate)(
                            x, deterministic=not training)
        return x


# TODO: Replace with TrainState when it's ready
# https://github.com/google/flax/blob/master/docs/flip/1009-optimizer-api.md#train-state
@flax.struct.dataclass
class Model:
    step: int
    apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None
               ) -> 'Model':
        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def.apply,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn({'params': self.params}, *args, **kwargs)

    def apply_gradient(
            self,
            loss_fn: Optional[Callable[[Params], Any]] = None,
            grads: Optional[Any] = None,
            has_aux: bool = True) -> Union[Tuple['Model', Any], 'Model']:

        if grads is None:
            grad_fn = jax.grad(loss_fn, has_aux=has_aux)
            if has_aux:
                _grads, aux = grad_fn(self.params)
            else:
                _grads = grad_fn(self.params)
        else:
            _grads = grads

        updates, new_opt_state = self.tx.update(_grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1,
                                 params=new_params,
                                 opt_state=new_opt_state)
        if has_aux:
            return new_model, aux
        else:
            return new_model

    def update_params(self, new_params: Params) -> 'Model':
        return self.replace(params=new_params)
    
    def reset_optimizer(self) -> 'Model':
        # reset optimizer for the next task 
        # skip the count argument.
        # adam_state = self.opt_state[1][0]
        # mu = adam_state.mu
        # mu = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), mu)
        # nu = adam_state.nu
        # nu = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), nu)
        # init_opt_state = (self.opt_state[0], (
        #     adam_state._replace(mu=mu, nu=nu), self.opt_state[1][1:])
        # )
        
        # contain the count argument
        init_opt_state = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x),
                                                self.opt_state)
        return self.replace(opt_state=init_opt_state)

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)


@flax.struct.dataclass
class AlterTrainableModel:
    step: int
    apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
    params: Params
    optimizer_one: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    optimizer_two: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state_one: Optional[optax.OptState] = None
    opt_state_two: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               params_list_one: list = None,
               params_list_two: list = None,
               optimizer_one: Optional[optax.GradientTransformation] = None,
               optimizer_two: Optional[optax.GradientTransformation] = None,
               ) -> 'AlterTrainableModel':
        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        # the first set of trainable params
        tx_one = optax.multi_transform({'train': optimizer_one, 'fix': optax.set_to_zero()},
            create_mask(params, params_list_one))
        opt_state_one = tx_one.init(params)

        # another set of trainable param
        tx_two = optax.multi_transform({'train': optimizer_two, 'fix': optax.set_to_zero()},
            create_mask(params, params_list_two))
        opt_state_two = tx_two.init(params)

        return cls(step=1,
                   apply_fn=model_def.apply,
                   params=params,
                   optimizer_one=tx_one,
                   optimizer_two=tx_two,
                   opt_state_one=opt_state_one,
                   opt_state_two=opt_state_two)

    def __call__(self, *args, **kwargs):
        return self.apply_fn({'params': self.params}, *args, **kwargs)

    def apply_gradient_one(
            self,
            loss_fn: Optional[Callable[[Params], Any]] = None,
            grads: Optional[Any] = None,
            has_aux: bool = True
            ) -> Union[Tuple['AlterTrainableModel', Any], 'AlterTrainableModel']:

        if grads is None:
            grad_fn = jax.grad(loss_fn, has_aux=has_aux)
            if has_aux:
                _grads, aux = grad_fn(self.params)
            else:
                _grads = grad_fn(self.params)
        else:
            _grads = grads

        updates, new_opt_state = self.optimizer_one.update(_grads, 
            self.opt_state_one, self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1,
                                 params=new_params,
                                 opt_state_one=new_opt_state)
        if has_aux:
            return new_model, aux
        else:
            return new_model

    def apply_gradient_two(
            self,
            loss_fn: Optional[Callable[[Params], Any]] = None,
            grads: Optional[Any] = None,
            has_aux: bool = True
            ) -> Union[Tuple['AlterTrainableModel', Any], 'AlterTrainableModel']:

        if grads is None:
            grad_fn = jax.grad(loss_fn, has_aux=has_aux)
            if has_aux:
                _grads, aux = grad_fn(self.params)
            else:
                _grads = grad_fn(self.params)
        else:
            _grads = grads

        updates, new_opt_state = self.optimizer_two.update(_grads, 
            self.opt_state_two, self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1,
                                 params=new_params,
                                 opt_state_two=new_opt_state)
        if has_aux:
            return new_model, aux
        else:
            return new_model

    def update_params(self, new_params: Params) -> 'AlterTrainableModel':
        return self.replace(params=new_params)
    
    def reset_optimizer(self) -> 'AlterTrainableModel':
        
        # contain the count argument
        opt_state_one = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x),
                                                self.opt_state_one)
        opt_state_two = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x),
                                                self.opt_state_two)
        return self.replace(opt_state_one=opt_state_one,
                            opt_state_two=opt_state_two)

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'AlterTrainableModel':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)


# TODO: Replace with TrainState when it's ready
# https://github.com/google/flax/blob/master/docs/flip/1009-optimizer-api.md#train-state
@flax.struct.dataclass
class ModelActor:
    step: int
    apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    ftd_tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None
    ftd_opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               components: jnp.ndarray,
               tx: Optional[optax.GradientTransformation] = None,
               ftd_tx: Optional[optax.GradientTransformation] = None) -> 'ModelActor':
        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        # inject the updated components into the params
        params = unfreeze(params)
        params['components'] = components
        params = freeze(params)

        opt_state = tx.init(params)
        
        # Freezes all but the finetuned layer.
        label_fn = flattened_traversal(
            lambda path, _: 'optimizer' if path[0] == 'codes' else 'none')
        ftd_tx = optax.multi_transform(
            {'optimizer': ftd_tx, 'none': optax.set_to_zero()}, label_fn)
        ftd_opt_state = ftd_tx.init(params.unfreeze())

        return cls(step=1,
                   apply_fn=model_def.apply,
                   params=params,
                   tx=tx,
                   ftd_tx=ftd_tx,
                   opt_state=opt_state,
                   ftd_opt_state=ftd_opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn({'params': self.params}, *args, **kwargs)

    def apply_gradient(
            self,
            loss_fn: Optional[Callable[[Params], Any]] = None,
            grads: Optional[Any] = None,
            has_aux: bool = True) -> Union[Tuple['ModelActor', Any], 'ModelActor']:

        if grads is None:
            grad_fn = jax.grad(loss_fn, has_aux=has_aux)
            if has_aux:
                grads, aux = grad_fn(self.params)
            else:
                grads = grad_fn(self.params)
        else:
            raise NotImplementedError

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1,
                                 params=new_params,
                                 opt_state=new_opt_state)
        if has_aux:
            return new_model, aux
        else:
            return new_model

    def finetune(
            self,
            loss_fn: Optional[Callable[[Params], Any]] = None,
            grads: Optional[Any] = None,
            has_aux: bool = True) -> Union[Tuple['ModelActor', Any], 'ModelActor']:

        if grads is None:
            grad_fn = jax.grad(loss_fn, has_aux=has_aux)
            if has_aux:
                grads, aux = grad_fn(self.params)
            else:
                grads = grad_fn(self.params)
        else:
            raise NotImplementedError

        updates, new_opt_state = self.ftd_tx.update(grads.unfreeze(), self.ftd_opt_state,
                                                    self.params.unfreeze())
        new_params = optax.apply_updates(self.params.unfreeze(), updates)

        new_model = self.replace(step=self.step + 1,
                                 params=freeze(new_params),
                                 ftd_opt_state=new_opt_state)
        if has_aux:
            return new_model, aux
        else:
            return new_model

    def update_params(self, unfreeze_params: Params) -> 'ModelActor':
        return self.replace(params=freeze(unfreeze_params))
    
    def reset_optimizer(self, tx: Optional[optax.GradientTransformation] = None, 
        ftd_tx: Optional[optax.GradientTransformation] = None) -> 'ModelActor':

        # Freezes all but the finetuned layer.
        label_fn = flattened_traversal(
            lambda path, _: 'optimizer' if path[0] == 'codes' else 'none')
        ftd_tx = optax.multi_transform(
            {'optimizer': ftd_tx, 'none': optax.set_to_zero()}, label_fn)

        # reset optimizer for the next task
        new_model = self.replace(
            step=1,
            tx=tx,
            ftd_tx=ftd_tx,
            opt_state=tx.init(self.params),
            ftd_opt_state=ftd_tx.init(self.params.unfreeze())
        )
        return new_model

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'ModelActor':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)