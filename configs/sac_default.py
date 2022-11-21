import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'sac'

    config.actor_lr = 1e-3
    config.critic_lr = 1e-3
    config.temp_lr = 1e-3

    config.hidden_dims = (256, 256, 256, 256)
    config.name_activation = 'leaky_relu'
    config.use_layer_norm = True

    config.discount = 0.99

    config.tau = 5e-3
    config.target_update_period = 1

    config.init_temperature = 1.0
    config.target_entropy = None
    config.backup_entropy = True

    config.replay_buffer_size = int(1e6)

    return config
