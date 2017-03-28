import nn

import tensorflow as tf
import numpy as np

from collections import deque
from copy import deepcopy

import gym
import math
import time

import threading

class history_object(object):
    def __init__(self, o, w):
        self.o = o
        self.w = w

class history(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.history = deque()

    def last(self, n):
        if n <= 0:
            return deque()

        start = 0
        if len(self.history) >= n:
            start = len(self.history) - n

        ret = deque()
        for i in range(start, len(self.history)):
            ret.append(self.history[i].o)
        
        return ret

    def size(self):
        return len(self.history)

    def append(self, e, w):
        qlen = len(self.history) + 1
        if qlen > self.max_size:
            for i in range(qlen - self.max_size):
                self.history.popleft()

        self.history.append(history_object(e, w))
        self.history = deque(sorted(self.history, key=lambda x: x.w))

    def sample(self, size):
        idx = []
        weights = []
        for i in range(len(self.history)):
            idx.append(i)
            weights.append(self.history[i].w)

        p_sum = float(sum(weights))
        p = []
        for w in weights:
            p.append(float(w) / p_sum)

        ret = deque()
        for i in range(min(size, len(p))):
            ch_idx = np.random.choice(idx, p=p)
            ret.append(self.history[ch_idx].o)
        
        return ret

class state(object):
    def __init__(self, shape, size):
        self.steps = deque()
        self.shape = shape
        self.size = size
        self.value = None

        for i in range(size):
            self.push_zeroes()

    def push_zeroes(self):
        self.push_array(np.zeros(self.shape))

    def push_array(self, step_array):
        assert self.shape == step_array.shape[0]

        if len(self.steps) == self.size:
            self.steps.popleft()

        self.steps.append(step_array)

    def complete(self):
        self.value = np.concatenate(self.steps)

    def read(self):
        return self.value

    def reshape(self, rows, cols):
        return self.value.reshape(rows, cols)

    def vector(self):
        return self.value.reshape(1, self.value.shape[0])

class gradient(object):
    def __init__(self, grad):
        self.grad = grad

    def update(self, v):
        self.grad += v

    def clear(self):
        self.grad = 0

    def read(self):
        return self.grad

class asyncq(object):
    def __init__(self, target, main, input_size, output_size, swriter):
        self.steps = 0

        self.actions = output_size

        self.gamma = 0.99
        self.alpha = 1.0

        self.target_update_step = 1000
        self.gradient_update_step = 100

        self.grads = {}

        self.random_action_alpha = 0.1
        self.ra_range_begin = 0.05
        self.random_action_alpha_cap = 1

        self.input_size = input_size

        self.target = target
        self.main = main

    def get_action(self, s):
        self.random_action_alpha = self.ra_range_begin + (self.random_action_alpha_cap - self.ra_range_begin) * math.exp(-0.0001 * self.steps)

        self.random_action_alpha = 0.1
        random_choice = np.random.choice([True, False], p=[self.random_action_alpha, 1-self.random_action_alpha])

        ch = 0
        if random_choice:
            ch = np.random.randint(0, self.actions)
        else:
            q = self.target.predict(s.vector())
            ch = np.argmax(q[0])

        return ch

    def step(self, st, action, r, nst, done):
        batch = [(st, action, r, nst, done)]
        self.batch_step(batch)

    def batch_step(self, batch):
        states_shape = (len(batch), self.input_size)
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

        qvals = self.target.predict(states)
        next_qvals = self.target.predict(next_states)

        for idx in range(len(batch)):
            e = batch[idx]
            s, a, r, sn, done = e

            qmax_next = np.amax(next_qvals[idx])
            if done:
                qmax_next = 0

            current_qa = qvals[idx][a]
            qsa = current_qa + self.alpha * (r + self.gamma * qmax_next - current_qa)
            qvals[idx][a] = qsa

        grads = self.main.compute_gradients(states, qvals)
        #loss, grads = self.main.train(states, qvals, True)
        if grads:
            for k, v in grads.iteritems():
                e = self.grads.get(k)
                if e:
                    e.update(v)
                else:
                    self.grads[k] = gradient(v)

        self.steps += 1

        if self.steps % self.target_update_step == 0 or done:
            self.target.import_params(self.main.export_params())

        if self.steps % self.gradient_update_step == 0 or done:
            self.main.apply_gradients(self.grads)
            for n, g in self.grads.iteritems():
                g.clear()

    def update_episode_stats(self, episodes, reward):
        self.main.update_episode_stats(episodes, reward)
        self.target.update_episode_stats(episodes, reward)

class run(object):
    def __init__(self, state_size, env):
        self.env = env

        self.total_steps = 0
        self.batch_size = 32

        self.steps = history(10)

        oshape = self.env.observation_space.shape
        self.current_state = state(oshape[0], state_size)

    def new_state(self, obs):
        self.current_state.push_array(obs)
        self.current_state.complete()
        return deepcopy(self.current_state)

    def do(self, get_action, step_callback, batch_callback):
        observation = self.env.reset()
        s = self.new_state(observation)

        done = False
        cr = 0
        while not done:
            #self.env.render()

            a = get_action(s)
            new_observation, reward, done, info = self.env.step(a)

            sn = self.new_state(new_observation)

            step_callback(s, a, reward, sn, done)

            self.steps.append((s, a, reward, sn, done), 1.0)

            #batch_callback(123)

            cr += reward
            self.total_steps += 1

            s = sn

        return cr

class mountain_car(object):
    def __init__(self, state_size, output_path):
        self.name = 'MountainCar-v0'

        self.state_size = state_size
        self.input_shape = 2
        self.output_size = 3
        self.input_size = self.state_size * self.input_shape

        output_path += '/run.%d' % (time.time())
        self.swriter = tf.summary.FileWriter(output_path)

        self.target = nn.nn("target", self.input_size, self.output_size, self.swriter)

    def run(self, aq, num_episodes):
        env = gym.make(self.name)

        oshape = env.observation_space.shape
        assert self.output_size == env.action_space.n
        assert oshape[0] == self.input_shape

        episodes = history(64)

        last_rewards = deque()
        last_rewards_size = 100

        total_steps = 0
        worse = 200.

        def batch_callback(unused):
            batch = []
            episodes_batch = episodes.sample(1)
            #episodes_batch = episodes.last(10)
            for e in episodes_batch:
                batch += e.steps.sample(64)

            if len(batch) > 0:
                aq.batch_step(batch)

        for i_episode in range(num_episodes):
            r = run(self.state_size, env)

            cr = r.do(aq.get_action, aq.step, batch_callback)

            k = worse/float(r.total_steps)
            k += (worse - r.total_steps)/worse

            episodes.append(r, k)

            aq.update_episode_stats(i_episode, cr)
            total_steps += r.total_steps

            if len(last_rewards) >= last_rewards_size:
                last_rewards.popleft()

            last_rewards.append(cr)
            mean = np.mean(last_rewards)
            std = np.std(last_rewards)

            print "%3d: reward: %3d, total steps: %d, mean reward over last %d episodes: %.1f, std: %.1f, k: %.2f" % (
                    i_episode, cr, total_steps, len(last_rewards), mean, std, k)

        env.close()

    def single_run(self, num_episodes):
        prefix = '%d' % (1)

        main = nn.nn(prefix + "_main", self.input_size, self.output_size, self.swriter)
        main.import_params(self.target.export_params())
        aq = asyncq(self.target, main, self.input_size, self.output_size, self.swriter)

        self.run(aq, num_episodes)

    def parallel_run(self, num_threads, num_episodes):

        threads = []
        for i in range(num_threads):
            prefix = '%d' % (i)

            main = nn.nn(prefix + "_main", self.input_size, self.output_size, self.swriter)
            main.import_params(self.target.export_params())

            aq = asyncq(self.target, main, self.input_size, self.output_size, self.swriter)

            r = threading.Thread(target=self.run, args=(aq, num_episodes,))
            r.start()
            threads.append(r)

        for r in threads:
            r.join()

if __name__ == '__main__':
    mc = mountain_car(10, 'mc0')
    mc.parallel_run(10, 10000)
    #mc.single_run(10000)
