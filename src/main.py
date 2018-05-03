#!/usr/bin/env python3
# Code from Repo SimonRamstedt/ddpg
# Heavily modified

import os
import pprint
import json

import gym
import numpy as np
import tensorflow as tf

import agent
import normalized_env
import runtime_env

import time
from tqdm import tqdm

import setproctitle
from datetime import datetime


srcDir = os.path.dirname(os.path.realpath(__file__))
rlDir = os.path.join(srcDir, 'archive')
plotScr = os.path.join(rlDir, 'plot-single.py')

cur_date_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('env', '', 'gym environment')
flags.DEFINE_string('outdir', 'output', 'output directory')
flags.DEFINE_string('exp_id', cur_date_time, 'experiment name')
flags.DEFINE_boolean('force', False, 'overwrite existing results')
flags.DEFINE_integer('train', 5000, 'training timesteps between testing episodes')
flags.DEFINE_integer('test', 5, 'testing episodes between training timesteps')
flags.DEFINE_integer('tmax', 999, 'maxium timesteps each episode')
flags.DEFINE_integer('total', 100000, 'total training timesteps')
flags.DEFINE_integer('total_episode', 100, 'total training episode')
flags.DEFINE_float('monitor', 0.01, 'probability of monitoring a test episode')
flags.DEFINE_string('model', 'ICNN', 'reinforcement learning model[DDPG, NAF, ICNN]')
flags.DEFINE_integer('tfseed', 0, 'random seed for tensorflow')
flags.DEFINE_integer('gymseed', 0, 'random seed for openai gym')
flags.DEFINE_integer('npseed', 0, 'random seed for numpy')
flags.DEFINE_float('ymin', 0, 'random seed for numpy')
flags.DEFINE_float('ymax', 1000, 'random seed for numpy')
flags.DEFINE_integer('n_trial', 5, 'number of trials')

setproctitle.setproctitle('ICNN.RL.{}.{}.{}'.format(
    FLAGS.env,FLAGS.model,FLAGS.tfseed))

os.makedirs(FLAGS.outdir, exist_ok=True)
#with open(os.path.join(FLAGS.outdir, 'flags.json'), 'w') as f:
    #json.dump(FLAGS.__flags, f, indent=2, sort_keys=True)

if FLAGS.model == 'DDPG':
    import ddpg
    Agent = ddpg.Agent
elif FLAGS.model == 'NAF':
    import naf
    Agent = naf.Agent
elif FLAGS.model == 'ICNN':
    import icnn
    Agent = icnn.Agent


class Experiment(object):
    def run(self, trial_i=0):
        self.train_timestep = 0
        self.test_timestep = 0
        self.episode = 1

        # create normal
        self.env = normalized_env.make_normalized_env(gym.make(FLAGS.env))
        tf.set_random_seed(FLAGS.tfseed)
        np.random.seed(FLAGS.npseed)
        #self.env.monitor.start(os.path.join(FLAGS.outdir, 'monitor'), force=FLAGS.force)
        self.env.seed(FLAGS.gymseed)
        #gym.logger.setLevel(gym.logging.WARNING)

        dimO = self.env.observation_space.shape
        dimA = self.env.action_space.shape
        pprint.pprint(self.env.spec.__dict__)

        self.agent = Agent(dimO, dimA=dimA)

        model_path = os.path.join(FLAGS.outdir, FLAGS.model)
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        test_log = open(os.path.join(model_path, 'test_{}.log'.format(trial_i)), 'w')
        train_log = open(os.path.join(model_path, 'train_{}.log'.format(trial_i)), 'w')

        #while self.train_timestep < FLAGS.total:
        while self.episode < FLAGS.total_episode:
            # test
            reward_list = []
            for _ in range(FLAGS.test):
                reward, timestep = self.run_episode(
                    test=True, monitor=np.random.rand() < FLAGS.monitor)
                reward_list.append(reward)
                self.test_timestep += timestep
            avg_reward = np.mean(reward_list)
            print('Average test return {} after {} timestep of training.'.format(
                avg_reward, self.train_timestep))
            #test_log.write("{}\t{}\n".format(self.train_timestep, avg_reward))
            test_log.write("{}\t{}\n".format(self.episode, avg_reward))
            test_log.flush()

            # train
            reward_list = []
            last_checkpoint = np.floor(self.train_timestep / FLAGS.train)
            while np.floor(self.train_timestep / FLAGS.train) == last_checkpoint:
                #print('=== Running episode')
                reward, timestep = self.run_episode(test=False, monitor=False)
                reward_list.append(reward)
                self.train_timestep += timestep
                #train_log.write("{}\t{}\n".format(self.train_timestep, reward))
                train_log.write("{}\t{}\n".format(self.episode, reward))
                train_log.flush()
            avg_reward = np.mean(reward_list)
            #print('Average train return {} after {} timestep of training.'.format(
                #avg_reward, self.train_timestep))

        #self.env.monitor.close()
        os.makedirs(os.path.join(model_path, "tf"), exist_ok=True)
        ckpt = os.path.join(model_path, "tf/model_{}.ckpt".format(trial_i))
        self.agent.saver.save(self.agent.sess, ckpt)

    def run_episode(self, test=True, monitor=False):
        #self.env.monitor.configure(lambda _: monitor)
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

            filtered_reward = self.env.filter_reward(reward)

            if reward > 0.0:
                print("reached the goal after {} steps with reward {}".format(timestep, reward))

            start = time.clock()
            self.agent.observe(filtered_reward, term, observation, test=test)
            times['obs'].append(time.clock()-start)

            sum_reward += reward
            timestep += 1

        #print('=== Episode stats:')
        for k,v in sorted(times.items()):
            #print('  + Total {} time: {:.4f} seconds'.format(k, np.mean(v)))
            pass

        #print('  + Reward: {}'.format(sum_reward))
        if not test:
            self.episode += 1
        return sum_reward, timestep

    def save_plot(self):
        os.system('{} --model_path {} --model {} --n_trial {}'.format(plotScr, self.model_path, FLAGS.model,
            FLAGS.n_trial))

def main():
    exp = Experiment()
    for i in tqdm(range(FLAGS.n_trial)):
        print("hi")
        with tf.Graph().as_default():
            exp.run(i)
        print("bye")
    exp.save_plot()

if __name__ == '__main__':
    runtime_env.run(main, FLAGS.outdir)
