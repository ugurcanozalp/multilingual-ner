import sys
from argparse import ArgumentParser
import os
import torch
from multiner.plmodule import MultiNerTrainer

parser = ArgumentParser()
parser.add_argument('--ckpt_file', type=str, default="lightning_logs/version_0/checkpoints/epoch=3-step=3.ckpt")
parser.add_argument('--tags_file', type=str, default="data/tags.txt")
parser.add_argument('--name', type=str, default="custom")
parser.add_argument('--output_folder', type=str, default="model_store")
args = parser.parse_args()

plmodule = MultiNerTrainer(tags_path=args.tags_file)
plmodule.load_state_dict(torch.load(args.ckpt_file)['state_dict'])

model_dir = os.path.join(args.output_folder, args.name)
if not os.path.exists(model_dir):
	os.makedirs(model_dir)

torch.save(plmodule.model.state_dict(), os.path.join(model_dir, "model.pt"))
with open(os.path.join(model_dir, "tags.txt"), "w") as f:
	for tag in plmodule.tags:
		f.write(tag+'\n')
