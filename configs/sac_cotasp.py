import yaml
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.update_dict = True
    config.update_coef = True

    config.dict_configs = ml_collections.ConfigDict()
    config.dict_configs.c = 1.0                                   
    config.dict_configs.alpha = 1e-3                              
    config.dict_configs.method = 'lasso_lars'
    config.dict_configs.positive_code = False
    config.dict_configs.scale_code = False                              

    config.optim_configs = ml_collections.ConfigDict()
    config.optim_configs.lr = 3e-4                                
    config.optim_configs.max_norm = 1.0                           
    config.optim_configs.optim_algo = 'adam'                      
    config.optim_configs.clip_method = 'global_clip'              

    config.actor_configs = ml_collections.ConfigDict()
    config.actor_configs.hidden_dims = (1024, 1024, 1024)   
    config.actor_configs.name_activation = 'gelu'          
    config.actor_configs.use_rms_norm = False                     
    config.actor_configs.use_layer_norm = False                   
    config.actor_configs.final_fc_init_scale = 1e-3               
    config.actor_configs.clip_mean = 1.0                          
    config.actor_configs.state_dependent_std = True               

    config.critic_configs = ml_collections.ConfigDict()
    config.critic_configs.hidden_dims = (256, 256, 256)      
    config.critic_configs.name_activation = 'gelu'       
    config.critic_configs.use_layer_norm = False                  

    config.tau = 0.005
    config.init_temperature = 1.0                                 
    config.target_entropy = -2.0                                  # by default

    return config


if __name__ == "__main__":
    
    # kwargs = dict(get_config())
    # print(kwargs)
    yaml_dict = get_config().to_dict()
    with open('sac_cotasp.yaml', 'w') as file:
        yaml.dump(yaml_dict, file)

    # with open('sac_cotasp.yaml', 'r') as file:
    #     yaml_dict = yaml.unsafe_load(file)

    # yaml_dict = yaml.unsafe_load(get_config().to_yaml())
    # config = ml_collections.ConfigDict(yaml_dict)
    # print(dict(yaml_dict))
