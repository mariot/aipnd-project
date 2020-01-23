import json
import argparse
from utils import load_checkpoint, predict

parser = argparse.ArgumentParser(description='Use a trained network to predict the class for an input image.')
parser.add_argument("image_path", help="Path of the image input", type=str)
parser.add_argument("checkpoint", help="Trained heckpoint", type=str)
parser.add_argument("--top_k", help="Return top K most likely classes", type=int, default=3)
parser.add_argument("--category_names", help="A mapping of categories to real names", type=str, default="cat_to_name.json")
parser.add_argument("--gpu", help="Use GPU", action="store_true", default=False)
args = parser.parse_args()

model = load_checkpoint(args.checkpoint, args.gpu)
most_probs, top_labels = predict(args.image_path, model, args.gpu, args.top_k)

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

names = [cat_to_name[label] for label in top_labels]

print("Probable names: {}".format(names), "Class probability: {}".format(most_probs))
