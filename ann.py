import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l1_l2_regularizer

import numpy as np

RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

def get_param_name(s):
    return s.split('/')[1].split(':')[0]
def get_scope_name(s):
    return s.split('/')[0].split(':')[0]

class ann(object):
    def __init__(self, scope, input_size, output_size, summary_writer):
        print "going to initialize scope %s" % scope
        self.summary_writer = summary_writer
        self.scope = scope
        with tf.variable_scope(scope) as vscope:
            self.vscope = vscope
            self.do_init(input_size, output_size)
            print "scope %s has been initialized" % scope

    def init_neurons(self, input_layer, wname, wnum, bias_name=None):
        ishape = input_layer.get_shape()[1].value
        dims = [ishape, wnum]

        w = tf.get_variable(wname,
                initializer=tf.random_normal(dims),
                regularizer=l1_l2_regularizer(scale_l1=self.reg_beta, scale_l2=self.reg_beta),
                dtype=tf.float32)
        sw = tf.summary.histogram(wname, w)
        self.summary_weights.append(sw)

        self.transform_params[wname] = w
        wext = tf.placeholder(tf.float32, dims, name=wname+'_ext')
        w_transform_ops = w.assign(w * (1 - self.transform_lr) + wext * self.transform_lr)
        self.transform_ops.append(w_transform_ops)

        h = tf.matmul(input_layer, w)

        if bias_name:
            b = tf.get_variable(bias_name, initializer=tf.random_normal([wnum]), dtype=tf.float32)
            sb = tf.summary.histogram(bias_name, b)
            self.summary_weights.append(sb)

            self.transform_params[bias_name] = b

            bext = tf.placeholder(tf.float32, [wnum], name=bias_name+'_ext')

            b_transform_ops = b.assign(b * (1 - self.transform_lr) + bext * self.transform_lr)
            self.transform_ops.append(b_transform_ops)

            h = tf.add(h, b)

        return h

    def init_layer(self, input_layer, wname, wnum, nonlinear, bias_name=None):
        h = self.init_neurons(input_layer, wname, wnum, bias_name)
        if nonlinear:
            return nonlinear(h)
        return h

    def init_model(self, input_size, output_size):
        layers = [('w0', 256), ('w1', 64)]

        print "init_model scope: %s" % (tf.get_variable_scope().name)

        x = tf.placeholder(tf.float32, [None, input_size], name='x')
        action = tf.placeholder(tf.int32, [None, 1], name='action')
        reward = tf.placeholder(tf.float32, [None, 1], name='reward')

        self.add_summary(tf.summary.histogram('action', action))
        self.add_summary(tf.summary.histogram('reward', reward))

        input_dimension = input_size
        input_layer = x
        idx = 0
        for wname, wnum in layers:
            input_layer = self.init_layer(input_layer, wname, wnum, tf.nn.elu, 'b%d'%(idx))
            #input_layer = self.init_layer(input_layer, l, tf.nn.tanh)
            self.add_summary(tf.summary.histogram('h%d' % idx, input_layer))
            idx += 1

        self.policy = self.init_layer(input_layer, 'policy', output_size, tf.nn.softmax)
        self.value = self.init_layer(input_layer, 'value', 1, None)

        log_policy = tf.log(1e-6 + self.policy)
        actions = tf.one_hot(action, output_size)
        actions = tf.squeeze(actions, 1)

        log_probability_per_action = tf.reduce_sum(tf.multiply(log_policy, actions), axis=1)
        advantage = (reward - self.value)

        self.add_summary(tf.summary.scalar("advantage_mean", tf.reduce_mean(advantage)))
        self.add_summary(tf.summary.scalar("advantage_rms", tf.sqrt(tf.reduce_mean(tf.square(advantage)))))

        self.cost_policy = -log_probability_per_action * advantage
        self.add_summary(tf.summary.scalar("cost_policy_mean", tf.reduce_mean(self.cost_policy)))
        self.add_summary(tf.summary.scalar("cost_policy_rms", tf.sqrt(tf.reduce_mean(tf.square(self.cost_policy)))))

        self.cost_value = tf.square(self.value - reward)
        self.add_summary(tf.summary.scalar("cost_value_mean", tf.reduce_mean(self.cost_value)))
        self.add_summary(tf.summary.scalar("input_reward_mean", tf.reduce_mean(reward)))
        self.add_summary(tf.summary.scalar("value_mean", tf.reduce_mean(self.value)))

        #reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        #self.add_summary(tf.summary.scalar("reg_loss", reg_loss))

        #self.cost = tf.add_n([self.cost_value, reg_loss])
        #self.add_summary(tf.summary.scalar("cost", self.cost))
        #self.cost = self.cost_policy

    def add_summary(self, s):
        self.summary_all.append(s)

    def setup_gradients(self, prefix, opt, cost):
        grads = opt.compute_gradients(cost)
        ret_grads = []
        ret_names = []
        ret_apply = []

        for e in grads:
            grad, var = e

            if grad is None or var is None:
                continue

            #print "var: %s, gradient: %s" % (var, grad)
            if self.scope != get_scope_name(var.name):
                continue

            pname = get_param_name(var.name)
            gname = '%s/gradient_%s' % (prefix, pname)
            print "gradient %s -> %s" % (var, gname)


            # get all gradients
            ret_grads.append(grad)
            ret_names.append(gname)

            pl = tf.placeholder(tf.float32, shape=var.get_shape(), name=gname)
            clip = tf.clip_by_average_norm(pl, 1)
            ret_apply.append((clip, var))

            ag = tf.summary.histogram('%s/%s/apply_%s'% (self.scope, prefix, gname), clip)
            self.summary_apply_gradients.append(ag)

        return ret_grads, ret_names, ret_apply

    def do_init(self, input_size, output_size):
        self.learning_rate_start = 0.0025
        self.reg_beta_start = 0.001
        self.transform_lr_start = 1.0

        self.train_num = 0

        self.transform_ops = []
        self.transform_params = {}

        self.summary_all = []
        self.summary_weights = []
        self.episode_stats_update = []
        self.summary_apply_gradients = []

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.transform_lr = 0.00001 + tf.train.exponential_decay(self.transform_lr_start, global_step, 30000, 0.6, staircase=True)
        self.learning_rate = 0.00001 + tf.train.exponential_decay(self.learning_rate_start, global_step, 30000, 0.6, staircase=True)
        self.reg_beta = tf.train.exponential_decay(self.reg_beta_start, global_step, 30000, 0.6, staircase=True)

        self.add_summary(tf.summary.scalar('reg_beta', self.reg_beta))
        self.add_summary(tf.summary.scalar('transform_lr', self.transform_lr))
        self.add_summary(tf.summary.scalar('learning_rate', self.learning_rate))
        self.add_summary(tf.summary.scalar('global_step', global_step))

        episodes_passed_p = tf.placeholder(tf.int32, [], name='episodes_passed')
        episode_reward_p = tf.placeholder(tf.float32, [], name='episode_reward')

        self.episodes_passed = tf.get_variable('episodes_passed', [], initializer=tf.constant_initializer(0),
                trainable=False, dtype=tf.int32)
        self.episode_stats_update.append(self.episodes_passed.assign(episodes_passed_p))

        self.episode_reward = tf.get_variable('episode_reward', [], initializer=tf.constant_initializer(0), trainable=False)
        self.episode_stats_update.append(self.episode_reward.assign(episode_reward_p))

        self.add_summary(tf.summary.scalar('episodes_passed', self.episodes_passed))
        self.add_summary(tf.summary.scalar('episode_reward', self.episode_reward))

        self.init_model(input_size, output_size)

        opt = tf.train.RMSPropOptimizer(self.learning_rate,
                RMSPROP_DECAY,
                momentum=RMSPROP_MOMENTUM,
                epsilon=RMSPROP_EPSILON, name='optimizer')

        self.gradient_names_policy = []
        self.apply_grads_policy = []

        self.gradient_names_value = []
        self.apply_grads_value = []

        self.compute_gradients_step_policy, self.gradient_names_policy, self.apply_grads_policy = self.setup_gradients("policy", opt, self.cost_policy)

        self.compute_gradients_step_value, self.gradient_names_value, self.apply_grads_value = self.setup_gradients("value", opt, self.cost_value)

        apply_gradients = self.apply_grads_policy + self.apply_grads_value

        self.apply_gradients_step = opt.apply_gradients(apply_gradients, global_step=global_step)

        config=tf.ConfigProto(
                intra_op_parallelism_threads = 8,
                inter_op_parallelism_threads = 8,
            )
        self.sess = tf.Session(config=config)
        self.summary_weights_merged = tf.summary.merge(self.summary_weights)
        self.summary_merged = tf.summary.merge(self.summary_all)
        self.summary_apply_gradients_merged = tf.summary.merge(self.summary_apply_gradients)

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        self.sess.run(init)


    def update_gradients(self, states, dret, names, grads):
        for gname, grad in zip(names, grads):
            #value = np.sum(grad) / float(len(states))
            value = grad / float(len(states)) * 2.0

            g = dret.get(gname)
            if not g:
                dret[gname] = value
            else:
                g.update(value)
            #print "computed gradients %s, shape: %s" % (gname, grad.shape)
            #print grad

    def compute_gradients(self, states, action, reward):
        self.train_num += 1

        ops = [self.summary_merged, self.compute_gradients_step_policy, self.compute_gradients_step_value]
        summary, grads_policy, grads_value = self.sess.run(ops, feed_dict={
                self.scope + '/x:0': states,
                self.scope + '/action:0': action,
                self.scope + '/reward:0': reward,
            })
        self.summary_writer.add_summary(summary, self.train_num)

        dret = {}
        self.update_gradients(states, dret, self.gradient_names_policy, grads_policy)
        self.update_gradients(states, dret, self.gradient_names_value, grads_value)
        return dret


    def apply_gradients(self, grads):
        if len(grads) == 0:
            print "empty gradients to apply"
            return

        feed_dict = {}
        #print "apply: %s" % grads
        for n, g in grads.iteritems():
            gname = self.scope + '/' + n + ':0'
            #print "apply gradients to %s" % (gname)
            #print g
            feed_dict[gname] = g

        ops = [self.summary_weights_merged, self.summary_apply_gradients_merged, self.apply_gradients_step]
        summary_weights, summary_apply, grads = self.sess.run(ops, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary_weights, self.train_num)
        self.summary_writer.add_summary(summary_apply, self.train_num)

    def predict_policy(self, states):
        p = self.sess.run([self.policy], feed_dict={
                self.scope + '/x:0': states,
            })
        return p[0]
    def predict_value(self, states):
        p = self.sess.run([self.value], feed_dict={
                self.scope + '/x:0': states,
            })
        return p[0]
    def predict_both(self, states):
        p = self.sess.run([self.policy, self.value], feed_dict={
                self.scope + '/x:0': states,
            })
        return p

    def export_params(self):
        return self.sess.run(self.transform_params)

    def transform(self, x1, x2):
        lr = 0.9
        return x1 * lr + x2 * (1 - lr)

    def import_params(self, d):
        self.train_num += 1

        d1 = {}
        for k, v in d.iteritems():
            d1[self.scope + '/' + k + '_ext:0'] = v

        self.sess.run(self.transform_ops, feed_dict=d1)
        summary = self.sess.run([self.summary_weights_merged])
        self.summary_writer.add_summary(summary[0], self.train_num)

    def update_episode_stats(self, episodes, reward):
        summary = self.sess.run(self.episode_stats_update, feed_dict={
                self.scope + '/episodes_passed:0': episodes,
                self.scope + '/episode_reward:0': reward,
            })
