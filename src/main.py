#!/usr/bin/env python3
# Code from Repo SimonRamstedt/ddpg
# Heavily modified
# Code from icnn

import os
import pprint as pp
import json
import gym
import numpy as np
import tensorflow as tf
import time
import setproctitle

#import flags
import flags
flags = tf.app.flags
FLAGS = flags.FLAGS
from icnn.RL.src.icnn import Agent

srcDir = os.path.dirname(os.path.realpath(__file__))
rlDir = os.path.join(srcDir, '..')
plotScr = os.path.join(rlDir, 'plot-single.py')

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('env', '', 'gym environment')
flags.DEFINE_string('outdir', 'output', 'output directory')
flags.DEFINE_boolean('force', False, 'overwrite existing results')
flags.DEFINE_integer('train', 1000, 'training timesteps between testing episodes')
flags.DEFINE_integer('test', 1, 'testing episodes between training timesteps')
flags.DEFINE_integer('tmax', 1000, 'maxium timesteps each episode')
flags.DEFINE_integer('total', 100000, 'total training timesteps')
flags.DEFINE_float('monitor', 0.01, 'probability of monitoring a test episode')
flags.DEFINE_string('model', 'ICNN', 'reinforcement learning model[DDPG, NAF, ICNN]')
flags.DEFINE_integer('tfseed', 0, 'random seed for tensorflow')
flags.DEFINE_integer('gymseed', 0, 'random seed for openai gym')
flags.DEFINE_integer('npseed', 0, 'random seed for numpy')
flags.DEFINE_float('ymin', 0, 'random seed for numpy')
flags.DEFINE_float('ymax', 1000, 'random seed for numpy')

setproctitle.setproctitle('ICNN.RL.{}.{}.{}'.format(
    FLAGS.env,FLAGS.model,FLAGS.tfseed))

os.makedirs(FLAGS.outdir, exist_ok=True)
with open(os.path.join(FLAGS.outdir, 'flags.json'), 'w') as f:
    json.dump(FLAGS.__flags, f, indent=2, sort_keys=True)

#if FLAGS.model == 'DDPG':
#    import ddpg
#    Agent = ddpg.Agent
#elif FLAGS.model == 'NAF':
#    import naf
#    Agent = naf.Agent
#elif FLAGS.model == 'ICNN':




class Experiment(object):
    def __init__(self):
        self.env = gym.make('MountainCarContinuous-v0')
        self.dim_obs = self.env.observation_space.shape
        self.dim_action = self.env.action_space.shape

        self.sess = tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=FLAGS.thread,
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True)))

    def run(self):
        self.agent = Agent(self.dim_obs, dimA=self.dim_action)

        import pdb;pdb.set_trace()
        rewards = []
        for _ in range(10):
            reward, timestep =  self.run_episode(test=False, monitor=False)
            rewards.append(reward)


    def run_episode(self, test=True, monitor=False):
        observation = self.env.reset()
        self.agent.reset(observation)
        sum_reward = 0
        timestep = 0
        term = False
        times = {'act': [], 'envStep': [], 'obs': []}
        while not term:
            start = time.clock()
            action = self.agent.act(test=test)
            times['act'].append(time.clock()-start)

            start = time.clock()
            observation, reward, term, info = self.env.step(action)
            times['envStep'].append(time.clock()-start)
            term = (not test and timestep + 1 >= FLAGS.tmax) or term

            #filtered_reward = self.env.filter_reward(reward)
            filtered_reward = reward

            start = time.clock()
            self.agent.observe(filtered_reward, term, observation, test=test)
            times['obs'].append(time.clock()-start)

            sum_reward += reward
            timestep += 1

        print('=== Episode stats:')
        for k,v in sorted(times.items()):
            print('  + Total {} time: {:.4f} seconds'.format(k, np.mean(v)))

        print('  + Reward: {}'.format(sum_reward))
        return sum_reward, timestep


if __name__ == "__main__":
    """
    import icnn and setup for testing experiments
    """
    exp = Experiment()
    exp.run()


