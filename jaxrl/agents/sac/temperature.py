from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import InfoDict, TrainState


LOG_TEMP_MAX = 5.0
LOG_TEMP_MIN = -10.0

def temp_activation(temp, temp_min=LOG_TEMP_MIN, temp_max=LOG_TEMP_MAX):
    return temp_min + 0.5 * (temp_max - temp_min) * (jnp.tanh(temp) + 1.)


class Temperature(nn.Module):
    init_log_temp: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), self.init_log_temp))
        return jnp.exp(log_temp)


def update(temp: TrainState, entropy: float,
           target_entropy: float) -> Tuple[TrainState, InfoDict]:

    def temperature_loss_fn(temp_params):
        temperature = temp.apply_fn({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}

    grads_temp, info = jax.grad(temperature_loss_fn, has_aux=True)(temp.params)
    new_temp = temp.apply_gradients(grads=grads_temp)

    return new_temp, info
