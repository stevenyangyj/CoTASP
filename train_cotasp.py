'''
CONTINUAL TASK ALLOCATION IN META-POLICY NETWORK VIA SPARSE PROMPTING
'''

import itertools
import random
import time

import numpy as np
import wandb
import yaml
from absl import app, flags
from ml_collections import config_flags, ConfigDict

from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate_cl
from jaxrl.utils import Logger

from jaxrl.agents.sac.sac_learner import CoTASPLearner
from continual_world import TASK_SEQS, get_single_env

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'cw20', 'Environment name.')
flags.DEFINE_integer('seed', 68, 'Random seed.')
flags.DEFINE_string('base_algo', 'cotasp', 'base learning algorithm')

flags.DEFINE_string('env_type', 'random_init_all', 'The type of env is either deterministic or random_init_all')
flags.DEFINE_boolean('normalize_reward', True, 'Normalize rewards')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 200, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 20000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 128, 'Mini batch size.')
flags.DEFINE_integer('updates_per_step', 1, 'Gradient updating per # environment steps.')
flags.DEFINE_integer('buffer_size', int(1e6), 'Size of replay buffer')
flags.DEFINE_integer('max_step', int(1e6), 'Number of training steps for each task')
flags.DEFINE_integer('start_training', int(1e4), 'Number of training steps to start training.')
flags.DEFINE_integer('theta_step', int(990), 'Number of training steps for theta.')
flags.DEFINE_integer('alpha_step', int(10), 'Number of finetune steps for alpha.')

flags.DEFINE_boolean('rnd_explore', True, 'random policy distillation')
flags.DEFINE_integer('distill_steps', int(2e4), 'distillation steps')

flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_string('wandb_mode', 'disabled', 'Track experiments with Weights and Biases.')
flags.DEFINE_string('wandb_project_name', "CoTASP_ContiWorld", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', None, "the entity (team) of wandb's project")
flags.DEFINE_boolean('save_checkpoint', False, 'Save meta-policy network parameters')
flags.DEFINE_string('save_dir', '/home/yijunyan/Data/PyCode/CoTASP/logs', 'Logging dir.')

# YAML file path to cotasp's hyperparameter configuration
with open('configs/sac_cotasp.yaml', 'r') as file:
    yaml_dict = yaml.unsafe_load(file)
config_flags.DEFINE_config_dict(
    'config',
    ConfigDict(yaml_dict),
    'Training hyperparameter configuration.',
    lock_config=False
)

def main(_):
    # config tasks
    seq_tasks = TASK_SEQS[FLAGS.env_name]
    algo_kwargs = dict(FLAGS.config)
    algo = FLAGS.base_algo
    run_name = f"{FLAGS.env_name}__{algo}__{FLAGS.seed}__{int(time.time())}"

    if FLAGS.save_checkpoint:
        save_policy_dir = f"logs/saved_actors/{run_name}.json"
        save_dict_dir = f"logs/saved_dicts/{run_name}"
    else:
        save_policy_dir = None
        save_dict_dir = None

    wandb.init(
        project=FLAGS.wandb_project_name,
        entity=FLAGS.wandb_entity,
        sync_tensorboard=True,
        config=FLAGS,
        name=run_name,
        monitor_gym=False,
        save_code=False,
        mode=FLAGS.wandb_mode,
        dir=FLAGS.save_dir
    )
    wandb.config.update({"algo": algo})

    log = Logger(wandb.run.dir)

    # random numpy seeding
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # initialize SAC agent
    temp_env = get_single_env(
        TASK_SEQS[FLAGS.env_name][0]['task'], FLAGS.seed, 
        randomization=FLAGS.env_type)
    if algo == 'cotasp':
        agent = CoTASPLearner(
            FLAGS.seed,
            temp_env.observation_space.sample()[np.newaxis],
            temp_env.action_space.sample()[np.newaxis], 
            len(seq_tasks),
            **algo_kwargs)
        del temp_env
    else:
        raise NotImplementedError()
    
    '''
    continual learning loop
    '''
    eval_envs = []
    for idx, dict_task in enumerate(seq_tasks):
        eval_envs.append(get_single_env(dict_task['task'], FLAGS.seed, randomization=FLAGS.env_type))

    total_env_steps = 0
    for task_idx, dict_task in enumerate(seq_tasks):
        
        '''
        Learning subroutine for the current task
        '''
        print(f'Learning on task {task_idx+1}: {dict_task["task"]} for {FLAGS.max_step} steps')
        # start the current task
        agent.start_task(task_idx, dict_task["hint"])
        
        if task_idx > 0 and FLAGS.rnd_explore:
            '''
            (Optional) Rand policy distillation for better exploration in the initial stage
            '''
            for i in range(FLAGS.distill_steps):
                batch = replay_buffer.sample(FLAGS.batch_size)
                distill_info = agent.rand_net_distill(task_idx, batch)
                
                if i % (FLAGS.distill_steps // 10) == 0:
                    print(i, distill_info)
            # reset actor's optimizer
            agent.reset_actor_optimizer()

        # set continual world environment
        env = get_single_env(
            dict_task['task'], FLAGS.seed, randomization=FLAGS.env_type, 
            normalize_reward=FLAGS.normalize_reward
        )
        # reset replay buffer
        replay_buffer = ReplayBuffer(
            env.observation_space, env.action_space, FLAGS.buffer_size or FLAGS.max_step
        )
        # reset environment
        observation, done = env.reset(), False
        for idx, optimize_alpha in enumerate(
            itertools.islice(
                itertools.cycle([False]*FLAGS.theta_step + [True]*FLAGS.alpha_step), FLAGS.max_step
            )
        ):
            if idx < FLAGS.start_training:
                # initial exploration strategy proposed in ClonEX-SAC
                if task_idx == 0:
                    action = env.action_space.sample()
                else:
                    # uniform-previous strategy
                    mask_id = np.random.choice(task_idx)
                    action = agent.sample_actions(observation[np.newaxis], mask_id)
                    action = np.asarray(action, dtype=np.float32).flatten()
                
                # default initial exploration strategy
                # action = env.action_space.sample()
            else:
                action = agent.sample_actions(observation[np.newaxis], task_idx)
                action = np.asarray(action, dtype=np.float32).flatten()
                
            next_observation, reward, done, info = env.step(action)
            # counting total environment step
            total_env_steps += 1

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0
            # only for meta-world
            assert mask == 1.0

            replay_buffer.insert(
                observation, action, reward, mask, float(done), next_observation
            )
            
            # CRUCIAL step easy to overlook
            observation = next_observation

            if done:
                # EPISODIC ending
                observation, done = env.reset(), False
                for k, v in info['episode'].items():
                    wandb.log({f'training/{k}': v, 'global_steps': total_env_steps})

            if (idx >= FLAGS.start_training) and (idx % FLAGS.updates_per_step == 0):
                for _ in range(FLAGS.updates_per_step):
                    batch = replay_buffer.sample(FLAGS.batch_size)
                    update_info = agent.update(task_idx, batch, optimize_alpha)
                if idx % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        wandb.log({f'training/{k}': v, 'global_steps': total_env_steps})

            if idx % FLAGS.eval_interval == 0:
                eval_stats = evaluate_cl(agent, eval_envs, FLAGS.eval_episodes)

                for k, v in eval_stats.items():
                    wandb.log({f'evaluation/{k}': v, 'global_steps': total_env_steps})

                # Update the log with collected data
                eval_stats['cl_method'] = algo
                eval_stats['x'] = total_env_steps
                eval_stats['steps_per_task'] = FLAGS.max_step
                log.update(eval_stats)
    
        '''
        Updating miscellaneous things
        '''
        print('End of the current task')
        dict_stats = agent.end_task(task_idx, save_policy_dir, save_dict_dir)

    # save log data
    log.save()
    np.save(f'{wandb.run.dir}/dict_stats.npy', dict_stats)

if __name__ == '__main__':
    app.run(main)