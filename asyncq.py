import ann
import history
import state

import tensorflow as tf
import numpy as np

from collections import deque
from copy import deepcopy

import gym
import math
import time

import threading

class gradient(object):
    def __init__(self, grad):
        self.grad = grad

    def update(self, v):
        self.grad += v

    def clear(self):
        self.grad = 0

    def read(self):
        return self.grad

class main_solver(object):
    def __init__(self, name, state_size, output_path):
        self.name = name

        self.state_size = state_size
        self.input_shape = 2
        self.output_size = 3
        self.input_size = self.state_size * self.input_shape

        output_path += '/run.%d' % (time.time())
        self.swriter = tf.summary.FileWriter(output_path)

        self.network = ann.ann("main", self.input_size, self.output_size, self.swriter)

class runner(object):
    def __init__(self, suffix, name, main, num_episodes):
        self.total_steps = 0
        self.batch_size = 64
        self.gamma = 0.99

        self.gradient_update_step = 1
        self.main_update_step = 2

        self.grads = {}

        self.env = gym.make(name)

        oshape = self.env.observation_space.shape
        assert main.output_size == self.env.action_space.n
        assert oshape[0] == main.input_shape

        self.steps = history.history(self.batch_size)

        self.main = main
        self.num_episodes = num_episodes

        self.current_state = state.state(oshape[0], main.state_size)

        self.network = ann.ann('thread_' + suffix, main.input_size, main.output_size, main.swriter)
        self.network.import_params(main.network.export_params())

        self.thread = threading.Thread(target=self.run)

    def get_action(self, s):
        random_action_alpha = 0.5
        random_choice = np.random.choice([True, False], p=[random_action_alpha, 1-random_action_alpha])

        if random_choice:
            return np.random.randint(0, self.main.output_size)

        q = self.network.predict_policy(s.vector())
        return np.argmax(q[0])

    def new_state(self, obs):
        self.current_state.push_array(obs)
        self.current_state.complete()
        return deepcopy(self.current_state)

    def calc_grads(self, states, action, reward, done):
        grads = self.network.compute_gradients(states, action, reward)
        if grads:
            for k, v in grads.iteritems():
                e = self.grads.get(k)
                if e:
                    e.update(v)
                else:
                    self.grads[k] = gradient(v)

        self.total_steps += 1

        if self.total_steps % self.gradient_update_step == 0 or done:
            grads = {}
            for n, g in self.grads.iteritems():
                grads[n] = g.read()

            #self.main.apply_gradients(grads)
            self.network.apply_gradients(grads)
            for n, g in self.grads.iteritems():
                g.clear()

        if self.total_steps % self.main_update_step == 0 or done:
            self.main.network.import_params(self.network.export_params())

    def update_episode_stats(self, episodes, reward):
        self.main.network.update_episode_stats(episodes, reward)
        self.network.update_episode_stats(episodes, reward)


    def run_batch(self):
        batch = self.steps.last(self.steps.max_size)

        states_shape = (len(batch), self.main.input_size)
        states = np.zeros(shape=states_shape)
        new_states = np.zeros(shape=states_shape)

        idx = 0
        for e in batch:
            s, a, r, sn, done = e

            states[idx] = s.read()
            new_states[idx] = sn.read()
            idx += 1

        P = self.network.predict_policy(states)
        V = self.network.predict_value(new_states)

        rashape = (len(batch), 1)
        reward = np.zeros(shape=rashape)
        action = np.zeros(shape=rashape)

        prev_reward = 0

        have_done = False

        for _i in range(self.steps.size(), 0, -1):
            idx = _i - 1
            s, a, r, sn, done = self.steps.get(idx)

            if _i == self.steps.size():
                if not done:
                    prev_reward = V[idx]

            if done:
                have_done = True

            action[idx] = a
            reward[idx] = r + self.gamma * prev_reward
            prev_reward = reward[idx]

        self.calc_grads(states, action, reward, have_done)

    def run_episode(self):
        self.network.import_params(self.main.network.export_params())

        observation = self.env.reset()
        s = self.new_state(observation)

        done = False
        cr = 0
        steps = 0
        while not done:
            a = self.get_action(s)
            new_observation, reward, done, info = self.env.step(a)

            sn = self.new_state(new_observation)

            self.steps.append((s, a, reward, sn, done), 1.0)

            if self.steps.full() or done:
                self.run_batch()
                self.steps.clear()

            cr += reward
            steps += 1

            s = sn

        return steps, cr

    def run(self):
        last_rewards = deque()
        last_rewards_size = 100

        total_steps = 0
        worse = 200.

        for i in range(self.num_episodes):
            steps, cr = self.run_episode()

            k = worse/float(steps)
            k += (worse - steps)/worse

            #episodes.append(r, k)

            self.update_episode_stats(i, cr)
            total_steps += steps

            if len(last_rewards) >= last_rewards_size:
                last_rewards.popleft()

            last_rewards.append(cr)
            mean = np.mean(last_rewards)
            std = np.std(last_rewards)

            print "%4d: reward: %4d, total steps: %7d, mean reward over last %3d episodes: %.1f, std: %.1f, k: %.2f" % (
                    i, cr, total_steps, len(last_rewards), mean, std, k)


class mountain_car(object):
    def __init__(self, state_size, output_path):
        self.name = 'MountainCar-v0'
        self.main = main_solver(self.name, state_size, output_path)

    def parallel_run(self, num_threads, num_episodes):
        threads = []
        for i in range(num_threads):
            suffix = '%d' % (i)

            r = runner(suffix, self.name, self.main, num_episodes)
            threads.append(r)

            self.main.swriter.add_graph(r.network.sess.graph)

            r.thread.start()

        for r in threads:
            r.thread.join()

if __name__ == '__main__':
    mc = mountain_car(1, 'mc0')
    mc.parallel_run(1, 10000)
    #mc.single_run(10000)
