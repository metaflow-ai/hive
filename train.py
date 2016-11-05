import os, argparse, json

from bee_simple import Bee_simple

dir = os.path.dirname(os.path.realpath(__file__))

print('Loading datasets')
with open(dir + '/dataset/data.json') as data_file:    
    data = json.load(data_file)
    train_data = data['train_data']
    dev_data = data['dev_data']

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", default=10, type=int, help="How many epochs should we train the GloVe (default: %(default)s)")
parser.add_argument("--batch_size", default=32, type=int, help="How many epochs should we train the GloVe (default: %(default)s)")
parser.add_argument("--eval_freq", default=10, type=int, help="How many epochs should we train the GloVe (default: %(default)s)")
parser.add_argument("--save_freq", default=251, type=int, help="How many epochs should we train the GloVe (default: %(default)s)")
args = parser.parse_args()

model = Bee_simple()
model.fit(args, train_data, dev_data)
