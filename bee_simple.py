import os, time, argparse, json

import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize

dir = os.path.dirname(os.path.realpath(__file__))

print('Building bee graph')
with tf.variable_scope("placeholder"):
    x_plh = tf.placeholder(tf.int32, shape=[None, 224, 224, 3])
    y_plh = tf.placeholder(tf.int32, shape=[None, 1])
    y_true_reshaped = tf.reshape(y_plh, [-1])

# First compresse channel wise
with tf.variable_scope("bee_conv"):
    W1 = tf.get_variable('W1', shape=[3, 3, 3, 32], initializer=tf.random_normal_initializer(stddev=1e-1))
    b1 = tf.get_variable('b1', shape=[32], initializer=tf.constant_initializer(0.1))
    z1 = tf.nn.conv2d(x_plh, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
    a = tf.nn.relu(z1)

with tf.variable_scope("classifier"):
    shape = a.get_shape().as_list()
    a_vec_size = shape[1] * shape[2] * shape[3]
    a_vec = tf.reshape(a, [-1, a_vec_size])

    W_fc1 = tf.get_variable('W_fc1', shape=[a_vec_size, 100], initializer=tf.random_normal_initializer(stddev=1e-1))
    b_fc1 = tf.get_variable('b_fc1', shape=[100], initializer=tf.constant_initializer(0.1))
    z = tf.matmul(a_vec, W_fc1) + b_fc1
    a = tf.nn.relu(z)

    W_fc2 = tf.get_variable('W_fc2', shape=[100, 2], initializer=tf.random_normal_initializer(stddev=1e-1))
    b_fc2 = tf.get_variable('b_fc2', shape=[2], initializer=tf.constant_initializer(0.1))
    a = tf.matmul(a, W_fc2) + b_fc2

with tf.variable_scope('loss'):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(a, y_true_reshaped)
    total_loss = tf.reduce_mean(losses)
    tf.scalar_summary("Loss", total_loss)

global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
adam = tf.train.AdamOptimizer(learning_rate=1e-3)
train_op = adam.minimize(total_loss, global_step=global_step)

train_summaries_op = tf.merge_all_summaries()

with tf.variable_scope('Accuracy'):
    preds = tf.cast(tf.argmax(a, 1, name="predictions"), tf.int32)
    correct_predictions = tf.equal(preds, y_true_reshaped)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
    acc_summary = tf.scalar_summary("Accuracy", accuracy)

result_folder = dir + '/results/' + str(int(time.time()))

def train_step(sess, x_batch, y_batch):
    to_compute = [train_op, train_summaries_op, total_loss, global_step]
    return sess.run(to_compute, feed_dict={
        x_plh: x_batch,
        y_plh: y_batch
    })

def dev_step(sess, x_batch, y_batch):
    to_compute = [accuracy, acc_summary]
    return sess.run(to_compute, feed_dict={
        x_plh: x_batch,
        y_plh: y_batch
    })

def preprocess_img(img_url):
    img = imread(img_url, mode='RGB')
    return imresize(img, (224, 224))

def preprocess(data):
    data = map(lambda val: (preprocess_img(val[0]), [val[1]]), data)
    x, y = zip(*data)
    x = np.array(x)
    y = np.array(y)
    return x, y

def ptb_iterator(raw_data, batch_size):
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = []
    for i in range(batch_len):
        data.append(raw_data[batch_size * i:batch_size * (i + 1)])

    epoch_size = (batch_len - 1)
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size")

    # print('epoch_size: %d, data_len: %d' % (epoch_size, len(data)))
    for i in range(epoch_size):
        x, y = preprocess(data[i])
        print(x.shape,y.shape)
        yield x, y

print('Loading datasets')
with open(dir + '/dataset/data.json') as data_file:    
    data = json.load(data_file)
    train_data = data['train_data']
    x_dev_batch, y_dev_batch = preprocess(data['dev_data'])
    print(x_dev_batch.shape, y_dev_batch.shape)
    x_test_batch, y_test_batch = preprocess(data['test_data'])
    print(x_test_batch.shape, y_test_batch.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", default=1, type=int, help="How many epochs should we train the GloVe (default: %(default)s)")
    parser.add_argument("--batch_size", default=1, type=int, help="How many epochs should we train the GloVe (default: %(default)s)")
    parser.add_argument("--eval_freq", default=100, type=int, help="How many epochs should we train the GloVe (default: %(default)s)")
    parser.add_argument("--save_freq", default=251, type=int, help="How many epochs should we train the GloVe (default: %(default)s)")
    args = parser.parse_args()

    bee_saver = tf.train.Saver()
    with tf.Session() as sess:
        print("Init models")
        sess.run(tf.initialize_all_variables())

        print('Bookeeping funcs')
        sw = tf.train.SummaryWriter(result_folder, sess.graph)

        for i in range(args.num_epochs):
            train_iterator = ptb_iterator(train_data, args.batch_size)
            for x_batch, y_batch in train_iterator:
                _, train_summaries, total_loss, current_step = train_step(sess, x_batch, y_batch)
                sw.add_summary(train_summaries, current_step)

                if current_step % args.eval_freq == 0:
                    dev_summaries = dev_step(sess, x_dev_batch, y_dev_batch)
                    sw.add_summary(dev_summaries, current_step)

                if current_step % args.save_freq == 0:
                    bee_saver.save(sess, result_folder + '/bee.chkp', global_step=current_step)
            epoch_acc = dev_step(sess, x_dev_batch, y_dev_batch)
            print('Epoch: %d, Accuracy: %f' % (i + 1, epoch_acc))

        bee_saver.save(sess)
