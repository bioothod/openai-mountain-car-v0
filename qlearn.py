import time
import math
import random

import numpy as np
import pandas as pd

import nn
import tensorflow as tf

class state(object):
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value

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

        output_path += '/run.%d' % (time.time())
        self.summary_writer = tf.summary.FileWriter(output_path)

        self.main = nn.nn("main", state_shape[0], actions, self.summary_writer)
        self.follower = nn.nn("follower", state_shape[0], actions, self.summary_writer)
        self.history = []

    def weighted_choice(self, ch):
        return np.random.choice()

    def get_action(self, state):
        self.total_actions += 1
        self.random_action_alpha = self.ra_range_begin + (self.random_action_alpha_cap - self.ra_range_begin) * math.exp(-0.0001 * self.total_actions)

        #self.random_action_alpha = 0.1
        random_choice = np.random.choice([True, False], p=[self.random_action_alpha, 1-self.random_action_alpha])

        ch = 0
        if random_choice:
            ch = np.random.randint(0, self.actions)
        else:
            v = state.value()
            q = self.follower.predict(v.reshape(1, v.shape[0]))
            ch = np.argmax(q[0])
            #print "state: %s, q: %s, action: %s" % (state, q, ch)

        return ch

    def truncate(self):
        diff = len(self.history) + 1 - self.history_size
        if diff > 0:
            self.history = self.history[diff:]

    def store(self, s, a, sn, r, done):
        self.truncate()
        self.history.append((s, a, sn, r, done))

    def learn(self):
        hsize = len(self.history)
        indexes = np.random.randint(hsize, size=min(self.batch_size, hsize))
        batch = []
        for i in indexes:
            batch.append(self.history[i])

        assert len(batch) != 0
        assert len(batch[0]) != 0
        assert len(batch[0][0].value()) != 0

        states_shape = (len(batch), len(batch[0][0].value()))
        states = np.ndarray(shape=states_shape)
        next_states = np.ndarray(shape=states_shape)

        q_shape = (len(batch), self.actions)
        qvals = np.ndarray(shape=q_shape)
        next_qvals = np.ndarray(shape=q_shape)

        idx = 0
        for e in batch:
            s, a, sn, r, done = e

            states[idx] = s.value()
            next_states[idx] = sn.value()
            idx += 1

        qvals = self.follower.predict(states)
        next_qvals = self.follower.predict(next_states)

        for idx in range(len(batch)):
            e = batch[idx]
            s, a, sn, r, done = e

            qmax_next = np.amax(next_qvals[idx])
            if done:
                qmax_next = 0

            current_qa = qvals[idx][a]
            qsa = current_qa + self.alpha * (r + self.gamma * qmax_next - current_qa)
            qvals[idx][a] = qsa

        self.main.train(states, qvals)

        if self.main.train_num % 10 == 0:
            self.follower.import_params(self.main.export_params())

    def remix(self, n, ft):
        total = len(self.history)

        if n > total:
            n = total
        start = total - n
        e = self.history[start:]
        b = self.history[0:start]

        mult = 100
        base = e * int(ft * mult)
        self.history = b + random.sample(base, int(n * ft))

        self.truncate()
        #print "total: %d, start: %d, history len: %d, x: %d" % (total, start, len(self.history), int(n*ft))

    def update_episode_stats(self, episodes, reward):
        self.main.update_episode_stats(episodes, reward)
        self.follower.update_episode_stats(episodes, reward)
