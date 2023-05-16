# Modified from https://github.com/hbzju/PiCO/blob/main/utils/cifar10.py

import os.path
import pickle
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from collections import OrderedDict

from .wide_resnet import WideResNet
from .utils_algo import generate_uniform_cv_candidate_labels, generate_instancedependent_candidate_labels
from .cutout import Cutout
from .autoaugment import CIFAR10Policy, ImageNetPolicy


def load_cifar10(args):
    
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    original_train = dsets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    ori_data, ori_labels = original_train.data, torch.Tensor(original_train.targets).long()
    
    test_dataset = dsets.CIFAR10(root=args.data_dir, train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size*4, shuffle=False, num_workers=args.workers, \
                                              sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))
    
    if args.exp_type == 'rand':
        partialY_matrix = generate_uniform_cv_candidate_labels(args, ori_labels)
    elif args.exp_type == 'ins':
        ori_data = torch.Tensor(original_train.data)
        model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
        model.load_state_dict(torch.load('./pmodel/cifar10.pt'))
        ori_data = ori_data.permute(0, 3, 1, 2)
        partialY_matrix = generate_instancedependent_candidate_labels(model, ori_data, ori_labels)
        ori_data = original_train.data
        
    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1
    
    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('Partial labels correctly loaded !')
    else:
        print('Inconsistent permutation !')

    print('Average candidate num: ', partialY_matrix.sum(1).mean())
    
    partial_training_dataset = CIFAR10_Partialize(ori_data, partialY_matrix.float(), ori_labels.float())
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(partial_training_dataset)
    
    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    
    return partial_training_dataloader, partialY_matrix, train_sampler, test_loader


class CIFAR10_Partialize(Dataset):
    def __init__(self, images, given_partial_label_matrix, true_labels):
        
        self.ori_images = images
        self.given_partial_label_matrix = given_partial_label_matrix
        self.true_labels = true_labels
        
        self.transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        
        self.transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        
        each_image1 = self.transform1(self.ori_images[index])
        each_image2 = self.transform2(self.ori_images[index])
        each_label = self.given_partial_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image1, each_image2, each_label, each_true_label, index

