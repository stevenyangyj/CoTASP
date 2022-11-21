from ast import Delete
import os
import random
import time

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from jaxrl.agents import SACLearner
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env

import continual_world as cw

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'cw10', 'Environment name.')
flags.DEFINE_string('save_dir', '/home/yijunyan/Data/PyCode/MORE/src/jaxrl/logs/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('base_algo', 'sac', 'base learning algorithm')

flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 128, 'Mini batch size.')
flags.DEFINE_integer('updates_per_step', 1, 'Gradient updates per step.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps for each task')
flags.DEFINE_integer('start_training', int(1e4), 'Number of training steps to start training.')

flags.DEFINE_integer('buffer_size', int(1e6), 'Size of replay buffer')
flags.DEFINE_boolean('reset_optimizers', True, 'Only reset the optimizers for pi, q, and temp')

flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_string('wandb_mode', 'online', 'Track experiments with Weights and Biases.')
flags.DEFINE_string('wandb_project_name', "jaxrl_demo", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', None, "the entity (team) of wandb's project")
config_flags.DEFINE_config_file(
    'config',
    'configs/sac_cw.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    kwargs = dict(FLAGS.config)
    algo = FLAGS.base_algo
    run_name = f"{FLAGS.env_name}_{algo}_{FLAGS.seed}_{int(time.time())}"  

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

    # summary_writer = SummaryWriter(
    #     os.path.join(FLAGS.save_dir, run_name))

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    # random numpy seeding
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # env = make_env(FLAGS.env_name, FLAGS.seed, video_train_folder)
    # eval_env = make_env(FLAGS.env_name, FLAGS.seed + 42, video_eval_folder)

    # initialize SAC agent
    temp_env = cw.get_single_env(
        cw.TASK_SEQS[FLAGS.env_name][0], 
        FLAGS.seed, randomization='deterministic')
    if algo == 'sac':
        agent = SACLearner(
            FLAGS.seed,
            temp_env.observation_space.sample()[np.newaxis],
            temp_env.action_space.sample()[np.newaxis], 
            **kwargs)
        del temp_env
    else:
        raise NotImplementedError()

    # continual learning loop
    list_task = cw.TASK_SEQS[FLAGS.env_name]
    total_env_steps = 0
    for idx, task in enumerate(list_task):
        print(f'Learning on task {idx+1}: {task} for {FLAGS.max_steps} steps')

        '''
        Learning subroutine for the current task
        '''
        # set continual world environment
        env = cw.get_single_env(task, FLAGS.seed, randomization='deterministic')
        eval_env = cw.get_single_env(task, FLAGS.seed, randomization='deterministic')

        # reset replay buffer
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                    FLAGS.buffer_size or FLAGS.max_steps)

        observation, done = env.reset(), False
        for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                        smoothing=0.1,
                        disable=not FLAGS.tqdm):
            if i < FLAGS.start_training:
                action = env.action_space.sample()
            else:
                action = agent.sample_actions(observation)
            next_observation, reward, done, info = env.step(action)
            # counting total environment step
            total_env_steps += 1

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0

            replay_buffer.insert(observation, action, reward, mask, float(done),
                                next_observation)
            observation = next_observation

            if done:
                observation, done = env.reset(), False
                for k, v in info['episode'].items():
                    wandb.log({f'training/{k}': v, 'global_steps': total_env_steps})
                    # summary_writer.add_scalar(f'training/{k}', v,
                    #                           info['total']['timesteps'])

                # if 'is_success' in info:
                #     wandb.log({})
                #     summary_writer.add_scalar(f'training/success',
                #                               info['is_success'],
                #                               info['total']['timesteps'])

            if i >= FLAGS.start_training:
                for _ in range(FLAGS.updates_per_step):
                    batch = replay_buffer.sample(FLAGS.batch_size)
                    update_info = agent.update(batch)

                if i % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        wandb.log({f'training/{k}': v, 'global_steps': total_env_steps})
                    #     summary_writer.add_scalar(f'training/{k}', v, i)
                    # summary_writer.flush()

            if i % FLAGS.eval_interval == 0:
                eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

                for k, v in eval_stats.items():
                    wandb.log({f'evaluation/average_{k}s': v, 'global_steps': total_env_steps})
                    # summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                    #                           info['total']['timesteps'])
                # summary_writer.flush()

                # eval_returns.append(
                #     (info['total']['timesteps'], eval_stats['return']))
                # np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                #            eval_returns,
                #            fmt=['%d', '%.1f'])
        
        '''
        Updating miscellaneous things
        '''
        if FLAGS.reset_optimizers:
            # reset agent for the next task
            print('Reset models for the next task')
            agent.reset_models()
            print('******************************')

if __name__ == '__main__':
    app.run(main)
