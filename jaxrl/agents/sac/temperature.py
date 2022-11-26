from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import InfoDict, Model


MAX_TEMP = 1e2
MIN_TEMP = 1e-4


class Temperature(nn.Module):
    init_temp: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.init_temp)))
        return jnp.exp(log_temp)


def update(temp: Model, entropy: float,
           target_entropy: float) -> Tuple[Model, InfoDict]:

    def temperature_loss_fn(temp_params):
        temperature = temp.apply_fn({'params': temp_params})
        # temperature = jnp.clip(temperature, MIN_TEMP, MAX_TEMP)
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}

    new_temp, info = temp.apply_gradient(temperature_loss_fn)

    return new_temp, info
