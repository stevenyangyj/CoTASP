import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 1e-3
    config.critic_lr = 1e-3
    config.temp_lr = 1e-3

    config.hidden_dims = (256, 256, 256, 256)
    config.name_activation = 'leaky_relu'
    config.use_layer_norm = False

    config.init_temperature = 1.0
    config.backup_entropy = True

    return config
