import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.sparsity_coeff = 1e-3

    config.optim_configs = ml_collections.ConfigDict()
    config.optim_configs.lr = 3e-4
    config.optim_configs.max_norm = 1.0
    config.optim_configs.optim_algo = 'adamw'
    config.optim_configs.clip_method = 'global_clip'
    config.optim_configs.decay_coef = 1e-5

    config.actor_configs = ml_collections.ConfigDict()
    config.actor_configs.hidden_dims = (1024, 1024, 1024, 1024)
    config.actor_configs.name_activation = 'leaky_relu'
    config.actor_configs.use_rms_norm = False
    config.actor_configs.use_layer_norm = False
    config.actor_configs.final_fc_init_scale = 1e-3

    config.critic_configs = ml_collections.ConfigDict()
    config.critic_configs.hidden_dims = (1024, 1024, 1024, 1024)
    config.critic_configs.name_activation = 'leaky_relu'
    config.critic_configs.use_layer_norm = False

    config.init_temperature = 1.0
    config.backup_entropy = True

    return config


if __name__ == "__main__":
    kwargs = dict(get_config())
    print(**kwargs)
