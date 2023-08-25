"""
Get data
"""

import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from dvsloader import get_cifar_loader, get_caltect_loader

def build_cifar(args, cutout=False, use_cifar10=True, download=True):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(root=args.data_path,
                                train=True, download=download, transform=transform_train)
        val_dataset = datasets.CIFAR10(root=args.data_path,
                              train=False, download=download, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = datasets.CIFAR100(root=args.data_path,
                                 train=True, download=download, transform=transform_train)
        val_dataset = datasets.CIFAR100(root=args.data_path,
                               train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset

def build_imagenet(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_dataset = datasets.ImageFolder(
        args.train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_dataset = datasets.ImageFolder(
        args.val_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    return train_dataset, val_dataset

def build_dvscifar(args):
    din = [48, 48]
    trainloader, testloader = get_cifar_loader(path=args.data_path, batch_size=args.batch_size, size=din[0])
    return trainloader, testloader

def build_ncaltech(args):
    din = [48, 48]
    trainloader, testloader = get_caltect_loader(path=args.data_path, batch_size=args.batch_size, size=din[0])
    return trainloader, testloader