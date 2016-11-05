import numpy as np
from scipy.misc import imread, imresize

def preprocess_img(img_url):
    img = imread(img_url, mode='RGB')
    return imresize(img, (64, 64))

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
        yield x, y