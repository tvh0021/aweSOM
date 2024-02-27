## This script is used to generate a parameter scan for the SOM. 
## Provided a list parameter values, it will print out the commands to be excecuted in a batch script.

import argparse

parser = argparse.ArgumentParser(description='generate a parameter scan for the SOM')
parser.add_argument("--script_location", type=str, dest='script_location', default='~/git_repos/som-plasma-3d/som3d.py')
parser.add_argument("--features_path", type=str, dest='features_path', default='/mnt/ceph/users/tha10/SOM-tests/hr-d3x640/')
parser.add_argument("--file", type=str, dest='file', default='features_4j1b1e_2800.h5')
parser.add_argument('--save_neuron_values', dest='save_neuron_values', action='store_true', help="Save the neuron values to a file")
parser.add_argument("--xdim", type=list[int], dest='xdim', default=[20, 22, 24])
parser.add_argument("--alpha", type=list[float], dest='alpha', default=[0.1])
parser.add_argument("--train", type=list[int], dest='train', default=[2000000, 4000000, 6000000, 8000000, 10000000])
parser.add_argument("--batch", type=list[int], dest='batch', default=[0])
args = parser.parse_args()

path = args.script_location
xdim = args.xdim
alpha = args.alpha
train = args.train
batch = args.batch
save_neuron_values = False
features_path = args.features_path
file = args.file

def print_excecution(path, xdim, alpha, train, batch, save_neuron_values, features_path, file):
    dim = xdim
    neuron_flag = ""
    if save_neuron_values:
        neuron_flag = "--save_neuron_values"

    batch_print = " --batch " + str(batch) if batch != 0 else ""
    print("python3 " + path + " --xdim " + str(dim) + " --ydim " + str(dim) + " --alpha " + str(alpha) + " --train " + str(train) + batch_print + neuron_flag + " --features_path '" + features_path + "' --file '" + file + "' &")

if __name__ == '__main__':
    for dim in xdim:
        for a in alpha:
            for steps in train:
                for window in batch:
                    print_excecution(path, dim, a, steps, window, save_neuron_values, features_path, file)

print("wait")
