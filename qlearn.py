import time
import math
import random

import numpy as np
import pandas as pd

import history
import nn
import state

import tensorflow as tf

class action(object):
    def __init__(self, n):
        self.action = np.array([n])

    def value(self):
        return self.action

    def __hash__(self):
        return hash(self.action)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return str(self.action)

class qlearn(object):
    def __init__(self, state_shape, actions, output_path):
        self.alpha = 1
        self.gamma = 0.99
        self.random_action_alpha = 1
        self.random_action_alpha_cap = 1
        self.ra_range_begin = 0.05
        self.ra_range_end = 0.99
        self.total_actions = 0
        self.lam = 0.9
        self.history_size = 100000
        self.batch_size = 256

        self.actions = actions
        self.history = history.history(self.history_size)

        output_path += '/run.%d' % (time.time())
        self.summary_writer = tf.summary.FileWriter(output_path)

        self.main = nn.nn("main", state_shape[0], actions, self.summary_writer)
        #self.follower = nn.nn("follower", state_shape[0], actions, self.summary_writer)

    def weighted_choice(self, ch):
        return np.random.choice()

    def get_action(self, s):
        self.total_actions += 1
        self.random_action_alpha = self.ra_range_begin + (self.random_action_alpha_cap - self.ra_range_begin) * math.exp(-0.0001 * self.total_actions)

        #self.random_action_alpha = 0.1
        random_choice = np.random.choice([True, False], p=[self.random_action_alpha, 1-self.random_action_alpha])

        if random_choice:
            return np.random.randint(0, self.actions)

        q = self.main.predict(s.vector())
        return np.argmax(q[0])

    def learn(self):
        batch = self.history.sample(min(self.batch_size, self.history.size()))

        assert len(batch) != 0
        assert len(batch[0]) != 0
        assert len(batch[0][0].read()) != 0

        states_shape = (len(batch), len(batch[0][0].read()))
        states = np.ndarray(shape=states_shape)
        next_states = np.ndarray(shape=states_shape)

        q_shape = (len(batch), self.actions)
        qvals = np.ndarray(shape=q_shape)
        next_qvals = np.ndarray(shape=q_shape)

        idx = 0
        for e in batch:
            s, a, r, sn, done = e

            states[idx] = s.read()
            next_states[idx] = sn.read()
            idx += 1

        qvals = self.main.predict(states)
        next_qvals = self.main.predict(next_states)

        for idx in range(len(batch)):
            e = batch[idx]
            s, a, r, sn, done = e

            qmax_next = np.amax(next_qvals[idx])
            if done:
                qmax_next = 0

            current_qa = qvals[idx][a]
            qsa = current_qa + self.alpha * (r + self.gamma * qmax_next - current_qa)
            qvals[idx][a] = qsa

        self.main.train(states, qvals)

        #if self.main.train_num % 10 == 0:
        #    self.follower.import_params(self.main.export_params())

    def update_episode_stats(self, episodes, reward):
        self.main.update_episode_stats(episodes, reward)
        #self.follower.update_episode_stats(episodes, reward)
