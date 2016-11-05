import os, json
from random import shuffle

# import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))

neg_folder = dir + '/dataset/neg'
pos_folder = dir + '/dataset/pos'
neg_files = [(neg_folder + '/' + f, 0) for f in os.listdir(neg_folder) if os.path.isfile(os.path.join(neg_folder, f))]
pos_files = [(pos_folder + '/' + f, 1) for f in os.listdir(pos_folder) if os.path.isfile(os.path.join(pos_folder, f))]
files = neg_files + pos_files

shuffle(files)
shuffle(files)
shuffle(files)
dev_data_percent = len(files) // 10

nb_files = len(files)
train_data = files[:nb_files - 2 * dev_data_percent]
dev_data = files[nb_files - 2 * dev_data_percent:nb_files - dev_data_percent]
test_data = files[nb_files - dev_data_percent:]
print(nb_files, len(train_data), len(dev_data), len(test_data))

with open(dir + '/dataset/data.json', 'w') as outfile:
    json.dump({
        'train_data': train_data,
        'dev_data': dev_data,
        'test_data': test_data
    }, outfile)

# neg_files_q = tf.train.string_input_producer(neg_files)
# pos_files_q = tf.train.string_input_producer(pos_files)

# reader = tf.WholeFileReader()
# key, value = reader.read(neg_files_q)

# my_img = tf.image.decode_jpeg(value, channels=3)
# img = tf.image.resize_images(my_img, (224, 224))

# print(img)