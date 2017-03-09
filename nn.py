import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l1_l2_regularizer

RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

class nn(object):
    def __init__(self, scope, input_size, output_size, summary_writer):
        self.summary_writer = summary_writer
        self.scope = scope
        with tf.variable_scope(scope):
            self.do_init(input_size, output_size)

    def init_layer(self, name_suffix, dims):
        wname = 'w' + name_suffix
        w = tf.get_variable(wname,
                initializer=tf.random_uniform(dims),
                regularizer=l1_l2_regularizer(scale_l1=self.reg_beta, scale_l2=self.reg_beta),
                dtype=tf.float32)
        sw = tf.summary.histogram(wname, w)
        self.summary_weights.append(sw)

        bname = 'b' + name_suffix
        b = tf.get_variable(bname, initializer=tf.random_uniform([dims[1]]), dtype=tf.float32)
        sb = tf.summary.histogram(bname, b)
        self.summary_weights.append(sb)

        wext = tf.placeholder(tf.float32, dims, name=wname+'_ext')
        bext = tf.placeholder(tf.float32, [dims[1]], name=bname+'_ext')

        w_transform_ops = w.assign(w * (1 - self.transform_lr) + wext * self.transform_lr)
        self.transform_ops.append(w_transform_ops)

        b_transform_ops = b.assign(b * (1 - self.transform_lr) + bext * self.transform_lr)
        self.transform_ops.append(b_transform_ops)

        self.transform_params.append(wname)
        self.transform_params.append(bname)

        return w, b

    def init_model(self, input_size, output_size):
        layers = [50, 190, output_size]
        #layers = [256, output_size]

        x = tf.placeholder(tf.float32, [None, input_size], name='x')
        y = tf.placeholder(tf.float32, [None, output_size], name='y')

        input_dimension = input_size
        input_layer = x
        idx = 0
        for l in layers:
            suffix = '%d' % (idx)
            w, b = self.init_layer(suffix, [input_dimension, l])

            hname = 'h%d' % (idx)
            h = tf.add(tf.matmul(input_layer, w), b, name=hname)

            if idx == len(layers) - 1:
                tf.summary.histogram('model', h)

                mse = tf.reduce_mean(tf.square(h - y))
                reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                loss = tf.add(mse, reg_loss, name='loss')
                tf.summary.scalar('mse', mse)
                tf.summary.scalar('loss', loss)
                return h, loss
            else:
                nl_h = tf.nn.tanh(h, name='nonlinear_' + hname)
                tf.summary.histogram('nonlinear_' + hname, nl_h)
                input_layer = nl_h

            input_dimension = l
            idx += 1


        return None, None


    def do_init(self, input_size, output_size):
        self.learning_rate_start = 0.0025
        self.reg_beta_start = 0.001
        self.transform_lr_start = 1.0

        self.train_num = 0

        self.transform_ops = []
        self.transform_params = []

        self.summary_weights = []
        self.episode_stats_update = []

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.transform_lr = 0.00001 + tf.train.exponential_decay(self.transform_lr_start, global_step, 30000, 0.6, staircase=True)
        self.learning_rate = 0.00001 + tf.train.exponential_decay(self.learning_rate_start, global_step, 30000, 0.6, staircase=True)
        self.reg_beta = tf.train.exponential_decay(self.reg_beta_start, global_step, 30000, 0.6, staircase=True)

        tf.summary.scalar('reg_beta', self.reg_beta)
        tf.summary.scalar('transform_lr', self.transform_lr)
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.scalar('global_step', global_step)

        episodes_passed_p = tf.placeholder(tf.int32, [], name='episodes_passed')
        episode_reward_p = tf.placeholder(tf.float32, [], name='episode_reward')

        self.episodes_passed = tf.get_variable('episodes_passed', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)
        self.episode_stats_update.append(self.episodes_passed.assign(episodes_passed_p))

        self.episode_reward = tf.get_variable('episode_reward', [], initializer=tf.constant_initializer(0), trainable=False)
        self.episode_stats_update.append(self.episode_reward.assign(episode_reward_p))

        tf.summary.scalar('episodes_passed', self.episodes_passed)
        tf.summary.scalar('episode_reward', self.episode_reward)

        self.model, self.loss = self.init_model(input_size, output_size)

        opt = tf.train.RMSPropOptimizer(self.learning_rate,
                RMSPROP_DECAY,
                momentum=RMSPROP_MOMENTUM,
                epsilon=RMSPROP_EPSILON, name='optimizer')

        self.optimizer_step = opt.minimize(self.loss, global_step=global_step)

        self.sess = tf.Session()
        self.summary_weights_merged = tf.summary.merge(self.summary_weights)
        self.merged = tf.summary.merge_all()

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self, states, qvals):
        self.train_num += 1

        summary, _, loss = self.sess.run([self.merged, self.optimizer_step, self.loss], feed_dict={
                self.scope + '/x:0': states,
                self.scope + '/y:0': qvals,
            })
        self.summary_writer.add_summary(summary, self.train_num)

        return loss

    def predict(self, states):
        p = self.sess.run([self.model], feed_dict={
                self.scope + '/x:0': states,
            })
        return p[0]

    def export_params(self):
        d = {}

        with tf.variable_scope(self.scope, reuse=True):
            for p in self.transform_params:
                d[p] = tf.get_variable(p)

        return self.sess.run(d)

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
