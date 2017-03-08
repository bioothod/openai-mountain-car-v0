import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l1_l2_regularizer

RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

class nn(object):
    def __init__(self, scope, input_size, output_size, summary_output):
        self.scope = scope
        with tf.variable_scope(scope):
            self.do_init(input_size, output_size, summary_output)

    def do_init(self, input_size, output_size, summary_output):
        self.params = ['w1', 'b1', 'w2', 'b2']

        self.learning_rate = 0.0025
        self.reg_beta = 0.001
        self.l1_neurons = 256
        self.train_num = 0
        self.transform_lr = 1.0

        self.transform_ops = []
        self.summary_weights = []

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        transform_lr = 0.00001 + tf.train.exponential_decay(self.transform_lr, global_step, 10000, 0.3, staircase=True)
        learning_rate = 0.00001 + tf.train.exponential_decay(self.learning_rate, global_step, 30000, 0.6, staircase=True)
        reg_beta = tf.train.exponential_decay(self.reg_beta, global_step, 30000, 0.6, staircase=True)
        tf.summary.scalar('global_step', global_step)

        x = tf.placeholder(tf.float32, [None, input_size], name='x')
        y = tf.placeholder(tf.float32, [None, output_size], name='y')

        w1 = tf.get_variable('w1',
                initializer=tf.random_uniform([input_size, self.l1_neurons]),
                regularizer=l1_l2_regularizer(scale_l1=reg_beta, scale_l2=reg_beta),
                dtype=tf.float32)
        sw1 = tf.summary.histogram('w1', w1)
        self.summary_weights.append(sw1)

        w1_ext = tf.placeholder(tf.float32, [input_size, self.l1_neurons], name='w1_ext')
        w1_transform_ops = w1.assign(w1 * (1 - transform_lr) + w1_ext * transform_lr)
        self.transform_ops.append(w1_transform_ops)

        b1 = tf.get_variable('b1', initializer=tf.random_uniform([self.l1_neurons]), dtype=tf.float32)
        sb1 = tf.summary.histogram('b1', b1)
        self.summary_weights.append(sb1)

        b1_ext = tf.placeholder(tf.float32, [self.l1_neurons], name='b1_ext')
        b1_transform_ops = b1.assign(b1 * (1 - transform_lr) + b1_ext * transform_lr)
        self.transform_ops.append(b1_transform_ops)

        h1 = tf.add(tf.matmul(x, w1), b1, name='h1')
        nl_h1 = tf.nn.tanh(h1, name='nonlinear_h1')
        tf.summary.histogram('nonlinear_h1', nl_h1)

        w2 = tf.get_variable('w2',
                initializer=tf.random_uniform([self.l1_neurons, output_size]),
                regularizer=l1_l2_regularizer(scale_l1=reg_beta, scale_l2=reg_beta),
                dtype=tf.float32)
        sw2 = tf.summary.histogram('w2', w2)
        self.summary_weights.append(sw2)

        w2_ext = tf.placeholder(tf.float32, [self.l1_neurons, output_size], name='w2_ext')
        w2_transform_ops = w2.assign(w2 * (1 - transform_lr) + w2_ext * transform_lr)
        self.transform_ops.append(w2_transform_ops)

        b2 = tf.get_variable('b2', initializer=tf.random_uniform([output_size]), dtype=tf.float32)
        sb2 = tf.summary.histogram('b2', b2)
        self.summary_weights.append(sb2)

        b2_ext = tf.placeholder(tf.float32, [output_size], name='b2_ext')
        b2_transform_ops = b2.assign(b2 * (1 - transform_lr) + b2_ext * transform_lr)
        self.transform_ops.append(b2_transform_ops)

        self.model =  tf.add(tf.matmul(nl_h1, w2), b2, name='model')
        tf.summary.histogram('model', self.model)

        a = tf.reduce_mean(tf.square(self.model - y))
        b = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.error = tf.add(a, b, name='error')
        tf.summary.scalar('error', self.error)

        opt = tf.train.RMSPropOptimizer(learning_rate,
                RMSPROP_DECAY,
                momentum=RMSPROP_MOMENTUM,
                epsilon=RMSPROP_EPSILON, name='optimizer')

        self.optimizer_step = opt.minimize(self.error, global_step=global_step)

        self.sess = tf.Session()
        self.summary_weights_merged = tf.summary.merge(self.summary_weights)
        self.merged = tf.summary.merge_all()
        self.summary_writter = tf.summary.FileWriter(summary_output, self.sess.graph)

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self, states, qvals):
        self.train_num += 1

        summary, _, error = self.sess.run([self.merged, self.optimizer_step, self.error], feed_dict={
                self.scope + '/x:0': states,
                self.scope + '/y:0': qvals,
            })
        #self.summary_writter.add_summary(summary, self.train_num)

        return error

    def predict(self, states):
        p = self.sess.run([self.model], feed_dict={
                self.scope + '/x:0': states,
            })
        return p[0]

    def export_params(self):
        d = {}

        with tf.variable_scope(self.scope, reuse=True):
            for p in self.params:
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
        #summary = self.sess.run([self.summary_weights_merged])
        #self.summary_writter.add_summary(summary[0], self.train_num)
