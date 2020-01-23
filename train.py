import argparse
from model import train

parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint.')
parser.add_argument("data_directory", help="Directory where the data is stored", type=str)
parser.add_argument("--save_dir", help="Directory to save checkpoints", type=str, default=".")
parser.add_argument("--arch", help="Architecture", type=str, default="vgg16")
parser.add_argument("--gpu", help="Use GPU", action="store_true", default=True)
parser.add_argument("--learning_rate", help="Learning rate", type=float, default=0.001)
parser.add_argument("--hidden_units", help="Hidden units", type=int, default=4096)
parser.add_argument("--epochs", help="Epochs", type=int, default=6)
args = parser.parse_args()

train(args.data_directory, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
