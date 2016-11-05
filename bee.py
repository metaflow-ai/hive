import os, time

import tensorflow as tf
import util

dir = os.path.dirname(os.path.realpath(__file__))

class Bee_vgg(object):
    def __init__(self):
        self.result_folder = dir + '/results/' + str(int(time.time()))
        self.graph = tf.Graph()

        print('Building bee graph')
        with self.graph.as_default():
            print('Loading vgg_graph')
            self.vgg_saver = tf.train.import_meta_graph(dir + '/vgg/results/vgg-16.meta')
            vgg_graph = tf.get_default_graph()

            self.x_plh = vgg_graph.get_tensor_by_name('input:0')
            conv5_3 =vgg_graph.get_tensor_by_name('conv5_3:0')

            print('Building bee graph')
            with tf.variable_scope("placeholder"):
                self.y_plh = tf.placeholder(tf.int32, shape=[None, 1])
                y_true_reshaped = tf.reshape(self.y_plh, [-1])

            # First compresse channel wise
            with tf.variable_scope("bee_conv"):
                W1 = tf.get_variable('W1', shape=[1, 1, 512, 32], initializer=tf.random_normal_initializer(stddev=1e-1))
                tf.add_to_collection("bee_vars", W1)
                b1 = tf.get_variable('b1', shape=[32], initializer=tf.constant_initializer(0.1))
                tf.add_to_collection("bee_vars", b1)

                z1 = tf.nn.conv2d(conv5_3, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
                a = tf.nn.relu(z1)

            with tf.variable_scope("classifier"):
                shape = a.get_shape().as_list()
                a_vec_size = shape[1] * shape[2] * shape[3]
                a_vec = tf.reshape(a, [-1, a_vec_size])

                W_fc1 = tf.get_variable('W_fc1', shape=[a_vec_size, 100], initializer=tf.random_normal_initializer(stddev=1e-1))
                tf.add_to_collection("bee_vars", W_fc1)
                b_fc1 = tf.get_variable('b_fc1', shape=[100], initializer=tf.constant_initializer(0.1))
                tf.add_to_collection("bee_vars", b_fc1)
                z = tf.matmul(a_vec, W_fc1) + b_fc1
                a = tf.nn.relu(z)

                W_fc2 = tf.get_variable('W_fc2', shape=[100, 2], initializer=tf.random_normal_initializer(stddev=1e-1))
                tf.add_to_collection("bee_vars", W_fc2)
                b_fc2 = tf.get_variable('b_fc2', shape=[2], initializer=tf.constant_initializer(0.1))
                tf.add_to_collection("bee_vars", b_fc2)
                z = tf.matmul(a, W_fc2) + b_fc2

            with tf.variable_scope('loss'):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(z, y_true_reshaped)
                self.total_loss = tf.reduce_mean(losses)
                tf.scalar_summary("Loss", self.total_loss)

            self.global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
            adam = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.train_op = adam.minimize(self.total_loss, global_step=self.global_step)

            self.train_summaries_op = tf.merge_all_summaries()

            with tf.variable_scope('Accuracy'):
                preds = tf.cast(tf.argmax(z, 1, name="predictions"), tf.int32)
                correct_predictions = tf.equal(preds, y_true_reshaped)
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
                self.acc_summary = tf.scalar_summary("Accuracy", self.accuracy)

            self.saver = tf.train.Saver()

    def train_step(self, sess, x_batch, y_batch):
        # print('train step', x_batch.shape, y_batch.shape)
        to_compute = [self.train_op, self.train_summaries_op, self.total_loss, self.global_step]
        return sess.run(to_compute, feed_dict={
            self.x_plh: x_batch,
            self.y_plh: y_batch
        })

    def dev_step(self, sess, x_batch, y_batch):
        # print('dev step', x_batch.shape, y_batch.shape)
        to_compute = [self.accuracy, self.acc_summary]
        return sess.run(to_compute, feed_dict={
            self.x_plh: x_batch,
            self.y_plh: y_batch
        })

    def fit(self, args, train_data, dev_data):
        x_dev_batch, y_dev_batch = util.preprocess(dev_data)
        with tf.Session(graph=self.graph) as sess:
            sw = tf.train.SummaryWriter(self.result_folder, sess.graph)

            print("Init models")
            sess.run(tf.initialize_all_variables())

            for i in range(args.num_epochs):
                train_iterator = util.ptb_iterator(train_data, args.batch_size)
                for x_batch, y_batch in train_iterator:
                    _, train_summaries, total_loss, current_step = self.train_step(sess, x_batch, y_batch)
                    sw.add_summary(train_summaries, current_step)

                    if current_step % args.eval_freq == 0:
                        acc, dev_summaries = self.dev_step(sess, x_dev_batch, y_dev_batch)
                        sw.add_summary(dev_summaries, current_step)

                    if current_step % args.save_freq == 0:
                        self.saver.save(sess, self.result_folder + '/bee.chkp', global_step=current_step)
                epoch_acc, dev_summaries = self.dev_step(sess, x_dev_batch, y_dev_batch)
                print('Epoch: %d, Accuracy: %f' % (i + 1, epoch_acc))

            self.saver.save(sess, self.result_folder + '/bee.chkp')

