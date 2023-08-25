"""
Setup dataset
"""
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from utils import str2bool
from dvsloader import DVSLoader, loadpt
from torchvision.datasets import DatasetFolder

parser = argparse.ArgumentParser(description='Data')
parser.add_argument('--dataset', type=str, help='dataset type')
parser.add_argument('--height', default=128, type=int, help='frame height')
parser.add_argument('--width', default=128, type=int, help='frame width')
parser.add_argument('--data_root', type=str, default='./data/', help='source file directory')
parser.add_argument('--out_path', type=str, default='./data/', help='output path of the .pt files')
parser.add_argument('--mode', type=str, default="", help='output path of the .pt files')
parser.add_argument('--T', default=16, type=int, metavar='N', help='snn simulation time (default: 16)')
parser.add_argument("--vis_fig", type=str2bool, nargs='?', const=True, default=False, help="visualize the event")
args = parser.parse_args()

def visualize(sample):
    for i in range(sample.shape[0]):
        fname = f"./{args.dataset}_{i}.png"
        x = sample[10].permute(1,2,0).numpy()

        plt.figure(figsize=(10,10))
        plt.imshow(x)
        plt.savefig(fname)
        plt.close()
        
def main():
    if not os.path.isdir(args.out_path):
        os.makedirs(args.out_path)

    mode = args.mode if not "cifar" in args.dataset else ""
    loader = DVSLoader(args.data_root, mode, height=args.height, width=args.width, save_dir=args.out_path)
    loader.event2quene(nframes=args.T, save=True, get_hist=False, dataset=args.dataset)

    # verify the dataset
    dataset = DatasetFolder(
        root=args.out_path,
        loader=loadpt,
        extensions=(".pt")
    )

    c = dataset.find_classes(args.out_path)
    sample, target = dataset.__getitem__(0)
    print(c[0])
    print("Sample shape = {}".format(sample.shape))
    print("Target label = {}".format(target))

    if args.vis_fig:
        visualize(sample)


if __name__ == '__main__':
    main()



    