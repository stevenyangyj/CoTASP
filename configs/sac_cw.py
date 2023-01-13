import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.optim_configs = ml_collections.ConfigDict()
    config.optim_configs.lr = 3e-4                                # [3e-4, 1e-3]
    config.optim_configs.max_norm = 1.0                           # [*1e3*, 1e4, 1e5, 1e6, 1e7]
    config.optim_configs.optim_algo = 'adam'                      # unadjustable
    config.optim_configs.clip_method = 'global_clip'              # unadjustable

    config.actor_configs = ml_collections.ConfigDict()
    config.actor_configs.hidden_dims = (256, 256, 256)         # unadjustable
    config.actor_configs.name_activation = 'leaky_relu'           # unadjustable
    config.actor_configs.use_rms_norm = False                     # unadjustable
    config.actor_configs.use_layer_norm = False                   # unadjustable
    config.actor_configs.final_fc_init_scale = 1e-3               # unadjustable
    config.actor_configs.clip_mean = 1.0                          # unadjustable
    config.actor_configs.state_dependent_std = True               # unadjustable

    config.critic_configs = ml_collections.ConfigDict()
    config.critic_configs.hidden_dims = (256, 256, 256)           # unadjustable
    config.critic_configs.name_activation = 'leaky_relu'          # unadjustable
    config.critic_configs.use_layer_norm = False                  # unadjustable

    config.tau = 0.005
    config.init_temperature = 1.0                                 # unadjustable
    config.target_entropy = -4.0                                  # by default

    return config
