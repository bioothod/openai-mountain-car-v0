import numpy as np

import gym
import time

import qlearn
import state

from copy import deepcopy

class mcar_pole:
    def __init__(self, num_episodes, output_path):
        self.num_episodes = num_episodes
        self.step = 0

        self.env = gym.make('MountainCar-v0')
        self.env = gym.wrappers.Monitor(self.env, 'mc0_wrappers')

        ospace = self.env.observation_space.shape

        self.obs_size = 2
        self.current_state = state.state(ospace[0], self.obs_size)

        self.q = qlearn.qlearn((ospace[0]*self.obs_size,), self.env.action_space.n, output_path)

    def new_state(self, obs):
        self.current_state.push_array(obs)
        self.current_state.complete()
        return deepcopy(self.current_state)

    def run(self):
        last_rewards = []
        last_rewards_size = 100

        for i_episode in range(self.num_episodes):
            observation = self.env.reset()
            s = self.new_state(observation)

            done = False
            cr = 0
            steps = 0
            while not done:
                #self.env.render()

                a = self.q.get_action(s)
                new_observation, reward, done, info = self.env.step(a)
                self.step += 1

                sn = self.new_state(new_observation)

                self.q.history.append((s, a, reward, sn, done), 1)
                self.q.learn()

                cr += reward
                steps += 1

                s = sn

            self.q.update_episode_stats(i_episode, cr)
            self.q.random_action_alpha_cap = self.q.ra_range_end - (self.q.ra_range_end - self.q.ra_range_begin) * (1. - steps/200.)

            if len(last_rewards) >= last_rewards_size:
                last_rewards = last_rewards[1:]

            last_rewards.append(cr)
            mean = np.mean(last_rewards)

            k = 200./float(steps)
            k += (200. - steps)/200.

            print "%d episode, its reward: %d, total steps: %d, mean reward over last %d episodes: %.1f, std: %.1f, k: %.2f" % (
                    i_episode, cr, self.step, len(last_rewards), mean, np.std(last_rewards), k)

        self.env.close()

import tensorflow as tf
with tf.device('/cpu:0'):
    cp = mcar_pole(10000, output_path='mc0')
    cp.run()
