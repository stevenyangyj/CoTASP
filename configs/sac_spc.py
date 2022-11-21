import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # config.coder_lr = 1e-3
    # config.actor_lr = 1e-3
    # config.critic_lr = 1e-3
    # config.temp_lr = 1e-3

    # config.global_norm = False
    # config.coder_norm = 1.0
    # config.actor_norm = 1.0
    # config.critic_norm = 1.0
    # config.temp_norm = 1.0

    config.optim_configs = ml_collections.ConfigDict()
    config.optim_configs.lr = 3e-4
    config.optim_configs.max_norm = 1.0
    config.optim_configs.optim_algo = 'adamw'
    config.optim_configs.clip_method = 'global_clip'
    config.optim_configs.decay_coef = 1e-5

    config.hidden_dims = (1024, 1024, 1024, 1024)
    config.name_activation = 'leaky_relu'
    config.use_layer_norm = False
    config.use_rms_norm = False

    config.s_warm_start = False
    config.s_start = 1/400.0
    config.s_end = 400.0

    config.component_nums = 10

    config.alpha = 0.75 # 0.02 (hat); 0.75 (spc)
    config.init_temperature = 1.0
    config.backup_entropy = True

    return config


if __name__ == "__main__":
    kwargs = dict(get_config())
    print(**kwargs)
