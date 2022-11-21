import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.coder_lr = 1e-3
    config.actor_lr = 1e-3
    config.critic_lr = 1e-3
    config.temp_lr = 1e-3

    config.hidden_dims = (256, 256, 256, 256)
    config.name_activation = 'leaky_relu'
    config.use_layer_norm = True

    config.init_temperature = 0.1
    config.backup_entropy = True

    return config
