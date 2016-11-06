import os, argparse

import tensorflow as tf
import freeze_graph as freeze_tools

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(model_folder):
    checkpoint = tf.train.get_checkpoint_state(args.model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    absolute_prefix_filename = input_checkpoint.split('.chkp')[0]
    input_graph = absolute_prefix_filename + '.pb'
    input_saver = absolute_prefix_filename + '_saver_def.pb'
    input_binary = True

    output_node_names = "Accuracy/predictions"
    restore_op_name = None
    filename_tensor_name = "save/Const:0"
    output_graph = absolute_model_folder + "/frozen_model.pb"
    clear_devices = True
    initializer_nodes = None
    freeze_tools.freeze_graph(
        input_graph, 
        input_saver,
        input_binary, 
        input_checkpoint,
        output_node_names, 
        restore_op_name,
        filename_tensor_name, 
        output_graph,
        clear_devices, 
        initializer_nodes
    )


parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", type=str, help="How many epochs should we train the GloVe (default: %(default)s)")
args = parser.parse_args()

freeze_graph(args.model_folder)