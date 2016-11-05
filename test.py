import os, argparse, json

from bee_simple import Bee_simple

dir = os.path.dirname(os.path.realpath(__file__))

print('Loading datasets')
with open(dir + '/dataset/data.json') as data_file:    
    data = json.load(data_file)
    test_data = data['test_data']

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", type=str, help="How many epochs should we train the GloVe (default: %(default)s)")
args = parser.parse_args()

model = Bee_simple()
model.eval(args, test_data)
