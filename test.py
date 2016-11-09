import os, time, argparse, json

from models.bee_simple import Bee_simple
from models.bee import Bee_vgg

dir = os.path.dirname(os.path.realpath(__file__))

print('Loading datasets')
with open(dir + '/dataset/data.json') as data_file:    
    data = json.load(data_file)
    test_data = data['test_data']

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", type=str, help="Model folder to export")
parser.add_argument("--model", default='simple', type=str, help="How many epochs should we train the GloVe (default: %(default)s)")
args = parser.parse_args()

start = time.clock()
if args.model == 'simple':
    model = Bee_simple()
else:
    model = Bee_vgg()
model.eval(args, test_data)
end = time.clock()
print('time spent predicting:', end - start)