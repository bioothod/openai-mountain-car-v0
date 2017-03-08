import numpy as np

import gym

import qlearn

class mcar_state(qlearn.state):
    def __init__(self, s):
        super(mcar_state, self).__init__(s)

    def __str__(self):
        return str(super(mcar_state, self).value())

class mcar_pole:
    def __init__(self, num_episodes):
        self.num_episodes = num_episodes
        self.step = 0

        self.env = gym.make('MountainCar-v0')

        ospace = self.env.observation_space.shape

        self.obs_size = 30
        self.obs_history = []
        for i in range(self.obs_size):
            self.obs_history.append(np.zeros(ospace[0]))

        self.q = qlearn.qlearn((ospace[0]*self.obs_size,), self.env.action_space.n)

    def new_state(self, obs):
        if len(self.obs_history) == self.obs_size:
            self.obs_history = self.obs_history[1:]
        self.obs_history.append(obs)
        return mcar_state(np.concatenate(self.obs_history))

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

                self.q.store(s, a, sn, reward, done)
                self.q.learn()

                cr += reward
                steps += 1

                s = sn

            self.q.random_action_alpha_cap = self.q.ra_range_end - (self.q.ra_range_end - self.q.ra_range_begin) * (1. - steps/200.)

            if len(last_rewards) >= last_rewards_size:
                last_rewards = last_rewards[1:]

            last_rewards.append(cr)
            mean = np.mean(last_rewards)

            k = 200./float(steps)
            k += (200. - steps)/200.

            print "%d episode, its reward: %d, total steps: %d, mean reward over last %d episodes: %.1f, std: %.1f, k: %.2f" % (
                    i_episode, cr, self.step, len(last_rewards), mean, np.std(last_rewards), k)


            self.q.remix(steps, k)

        self.env.close()

import tensorflow as tf
with tf.device('/cpu:0'):
    cp = mcar_pole(10000)
    cp.run()
