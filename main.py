import os
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import logging
from utils import str2bool, build_cifar, load_ddp_checkpoint, build_imagenet, build_dvscifar, build_ncaltech
from models import resnet19, svgg9
from trainers.trainer import STrainer
import csv
from models.layers import SConv
from models.cg import * 

parser = argparse.ArgumentParser(description='PyTorch SNN Training')
parser.add_argument('--model', type=str, help='model architecture')
parser.add_argument('--embedding', default=-1, type=int, help='Embedding size, -1 for plain adaptive pooling.')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 10)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD')
parser.add_argument('--wd', default=4e-5, type=float, help='weight decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--T', default=2, type=int, metavar='N', help='snn simulation time (default: 2)')
parser.add_argument('--means', default=1.0, type=float, metavar='N', help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--lamb', default=0.05, type=float, metavar='N', help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N', help='batch size')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
parser.add_argument('--lr_sch', type=str, default='cos', help='learning rate scheduler')

# dataset
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: CIFAR10 / ImageNet_1k')
parser.add_argument('--data_path', type=str, default='./data/', help='data directory')
parser.add_argument('--save_dir', type=str, default='./data/', help='data directory')

# data dir
parser.add_argument('--train_dir', type=str, default='./data/', help='training data directory')
parser.add_argument('--val_dir', type=str, default='./data/', help='validation data directory')
parser.add_argument('--lb', default=-3.0, type=float, help='lower bound')

# ddp
parser.add_argument('--seed', type=int, default=1000, help='use random seed to make sure all the processes has the same model')
parser.add_argument("--local_rank", type=int, help="Local rank [required]")

# save path
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--log_file', type=str, default=None, help='path to log file')

# amp training
parser.add_argument("--mixed_prec", type=str2bool, nargs='?', const=True, default=False, help="enable amp")

# Fine-tuning & Evaluate
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true', help='fine tuning from the pre-trained model, force the start epoch be zero')
parser.add_argument('--resume', default='', type=str, help='path of the pretrained model')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')

# low precision model
parser.add_argument('--wbit', type=int, default=4, help='activation precision')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    help='model architecture: ' +
                        ' (default: resnet18)')

args = parser.parse_args()

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

cnt_factor = 2
cnt_out = np.zeros(9 * cnt_factor) # this number is hardcoded for ResNet-20
cnt_full = np.zeros(9 * cnt_factor) # this number is hardcoded for ResNet-20
num_out = []
num_full = []

def _report_sparsity(m):
    classname = m.__class__.__name__
    if isinstance(m, CGConv2d):
        num_out.append(m.num_out)
        num_full.append(m.num_full)


def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    # initialize terminal logger
    logger = logging.getLogger('training')
    if args.log_file is not None:
        fileHandler = logging.FileHandler(args.save_path+args.log_file)
        fileHandler.setLevel(0)
        logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)
    logger.info(args)
    
    # unify the random seed for all the proesses 
    set_random_seeds(random_seed=args.seed)

    # initialize process
    torch.distributed.init_process_group(backend="nccl")
    args.world_size = torch.distributed.get_world_size()
    
    # data loader
    args.nprocs = 1 if args.evaluate else torch.cuda.device_count()
    args.batch_size = int(args.batch_size / args.nprocs)

    if args.dataset == "cifar10":
        train_dataset, val_dataset = build_cifar(args, use_cifar10=True)
        num_classes = 10
    elif args.dataset == "cifar100":
        train_dataset, val_dataset = build_cifar(args, use_cifar10=False)
        num_classes = 100
    elif args.dataset == "imagenet100":
        train_dataset, val_dataset = build_imagenet(args)
        num_classes = 100
    elif args.dataset == "dvscifar10":
        train_dataset, val_dataset = build_dvscifar(args)
        num_classes = 10
    elif args.dataset == "ncaltech101":
        train_dataset, val_dataset = build_ncaltech(args)
        num_classes = 101

    # Training sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.workers,
                                            pin_memory=True,
                                            sampler=train_sampler)

    # get model
    if args.model == "resnet19":
        model = resnet19(num_classes=num_classes)
    elif args.model == "vgg9":
        model = svgg9(num_classes=num_classes)

    model.T = args.T
    logger.info(model)
    

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    testloader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.workers,
                                            pin_memory=True,
                                            sampler=val_sampler)

    # get model
    if args.model == "resnet19":
        model = resnet19(num_classes=num_classes)
    elif args.model == "vgg9":
        model = svgg9(num_classes=num_classes)
    elif args.model == "mobilenet_tiny":
        model = mobilenet_tiny(num_classes=num_classes, wbit=args.wbit)
        
        # for the tapeout model 
        for m in model.modules():
            if hasattr(m, "levels"):
                m.interval = torch.tensor(0.125)
                ub, lb = 1.0, -2.0
                qrange = ub - lb

                m.levels = torch.tensor([lb + m.interval*i for i in range(int(qrange//m.interval))])
    else:
        raise NotImplementedError("Unsupported model architecture! (choice = resnet19, vgg, mobilenet_tiny)")
    
    model.T = args.T
    logger.info(model)
    args.fine_tune=False

    # resume from the checkpoint
    if args.fine_tune:
        logger.info("=> loading checkpoint...")
        state_tmp, load_acc = load_ddp_checkpoint(ckpt=args.resume, state=model.state_dict())
        model.load_state_dict(state_tmp)
        logger.info("=> loaded checkpoint! acc = {}%".format(load_acc))

    # initialize the trainer
    trainer = STrainer(
        model,
        trainloader,
        testloader,
        train_sampler,
        val_sampler,
        args,
        logger
    )

    if args.evaluate:
        trainer.valid_epoch() 
        logger.info("Test accuracy = {:.3f}".format(trainer.logger_dict["valid_top1"]))
        logger.info("FLOPS Reduction = {:.3f}".format(trainer.logger_dict["FLOPs_Reduction"]))
        exit()

    # start training
    trainer.fit()


if __name__ == '__main__':
    main()
