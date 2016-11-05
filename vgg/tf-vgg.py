import os 

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names

dir = os.path.dirname(os.path.realpath(__file__))

saver = tf.train.import_meta_graph(dir + '/results/vgg-16.meta')
graph = tf.get_default_graph()
probs = graph.get_tensor_by_name('preds:0')

if __name__ == '__main__':
    with tf.Session() as sess:
        print("Building the model")
        saver.restore(sess, dir + '/results/vgg-16')

        print('Preprocessing laska')
        img1 = imread('laska.png', mode='RGB')
        img1 = imresize(img1, (224, 224))

        print('Running the model')
        prob = sess.run(probs, feed_dict={'input:0': [img1]})[0]
        preds = (np.argsort(prob)[::-1])[0:5]
        print('Top five prediction:')
        for p in preds:
            print('    Prediction %s with proba %f' % (class_names[p], prob[p]))