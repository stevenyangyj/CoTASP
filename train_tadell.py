import random
import time

import numpy as np
import tqdm
import wandb
import yaml
from absl import app, flags
from ml_collections import config_flags, ConfigDict

from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate_cl
from jaxrl.utils import Logger

from jaxrl.agents.sac.sac_learner import TaDeLL
from continual_world import TASK_SEQS, get_single_env

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', "cw2-test", 'Environment name.')
flags.DEFINE_string('save_dir', '/home/yijunyan/Data/PyCode/CoTASP/logs', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 60, 'Random seed.')
flags.DEFINE_string('base_algo', 'tadell', 'base learning algorithm')

flags.DEFINE_boolean('ablation', False, 'Ablation study')
flags.DEFINE_boolean('save_checkpoint', False, 'Save meta-policy network parameters')

flags.DEFINE_string('env_type', 'deterministic', 'The type of env is either deterministic or random_init_all')
flags.DEFINE_boolean('normalize_reward', False, 'Normalize rewards')
flags.DEFINE_integer('eval_episodes', 1, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 20000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('updates_per_step', 1, 'Gradient updates per step.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps for each task')
flags.DEFINE_integer('start_training', int(1e4), 'Number of training steps to start training.')

flags.DEFINE_integer('buffer_size', int(1e6), 'Size of replay buffer')

flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_string('wandb_mode', 'online', 'Track experiments with Weights and Biases.')
flags.DEFINE_string('wandb_project_name', "jaxrl_tadell", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', None, "the entity (team) of wandb's project")
# YAML file path to tadell's hyperparameter configuration
with open('configs/sac_tadell.yaml', 'r') as file:
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

    kwargs = dict(FLAGS.config)
    algo = FLAGS.base_algo
    run_name = f"{FLAGS.env_name}__{algo}__{FLAGS.seed}__{int(time.time())}"

    if FLAGS.save_checkpoint:
        save_policy_dir = f"logs/saved_actors/{run_name}.json"
    else:
        save_policy_dir = None

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
    if algo == 'tadell':
        agent = TaDeLL(
            FLAGS.seed,
            temp_env.observation_space.sample()[np.newaxis],
            temp_env.action_space.sample()[np.newaxis], 
            **kwargs)
        del temp_env
    else:
        raise NotImplementedError()

    # continual learning loop
    eval_envs = []
    for task_idx, dict_task in enumerate(seq_tasks):
        # only for ablation study
        if task_idx == 0 and FLAGS.ablation:
            eval_seed = 60
        else:
            eval_seed = FLAGS.seed
        eval_envs.append(get_single_env(dict_task['task'], eval_seed, randomization=FLAGS.env_type))

    # continual learning loop
    total_env_steps = 0
    for task_idx, dict_task in enumerate(seq_tasks):
        print(f'Learning on task {task_idx+1}: {dict_task["task"]} for {FLAGS.max_steps} steps')

        '''
        Learning subroutine for the current task
        '''
        # start the current task
        agent.start_task(dict_task["hint"])

        # set continual world environment
        env = get_single_env(
            dict_task['task'], FLAGS.seed, randomization=FLAGS.env_type, 
            normalize_reward=FLAGS.normalize_reward
        )

        # reset replay buffer
        replay_buffer = ReplayBuffer(
            env.observation_space, env.action_space, FLAGS.buffer_size or FLAGS.max_steps
        )

        observation, done = env.reset(), False
        for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                        smoothing=0.1,
                        disable=not FLAGS.tqdm):
            if i < FLAGS.start_training:
                action = env.action_space.sample()
            else:
                action = agent.sample_actions(observation[np.newaxis])
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

            replay_buffer.insert(observation, action, reward, mask, float(done),
                                next_observation)
            observation = next_observation

            if done:
                observation, done = env.reset(), False
                for k, v in info['episode'].items():
                    wandb.log({f'training/{k}': v, 'global_steps': total_env_steps})

            if (i >= FLAGS.start_training) and (i % FLAGS.updates_per_step == 0):
                for _ in range(FLAGS.updates_per_step):
                    batch = replay_buffer.sample(FLAGS.batch_size)
                    update_info = agent.update(batch)

                if i % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        wandb.log({f'training/{k}': v, 'global_steps': total_env_steps})

            if i % FLAGS.eval_interval == 0:
                eval_stats = evaluate_cl(agent, eval_envs, FLAGS.eval_episodes, tadell=True)

                for k, v in eval_stats.items():
                    wandb.log({f'evaluation/average_{k}s': v, 'global_steps': total_env_steps})

                # Update the log with collected data
                eval_stats['cl_method'] = algo
                eval_stats['x'] = total_env_steps
                eval_stats['steps_per_task'] = FLAGS.max_steps
                log.update(eval_stats)
        
        '''
        Updating miscellaneous things
        '''
        print('End the current task')
        agent.end_task(save_policy_dir)

    # save log data
    log.save()

if __name__ == '__main__':
    app.run(main)