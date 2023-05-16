# Modified from https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

import os
import os.path
import sys
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from collections import OrderedDict

from .wide_resnet import WideResNet
from .utils_algo import generate_uniform_cv_candidate_labels, generate_instancedependent_candidate_labels
from .cutout import Cutout
from .autoaugment import CIFAR10Policy, ImageNetPolicy


def load_miniImagenet(args):
    
    test_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    original_train = miniImagenet(root=args.data_dir, train=True, transform=transforms.ToTensor())
    ori_data, ori_labels = original_train.data, torch.Tensor(original_train.targets).long()
    
    test_dataset = miniImagenet(root=args.data_dir, train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size*4, shuffle=False, \
    num_workers=args.workers, sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))
    
    
    if args.exp_type == 'rand':
        partialY_matrix = generate_uniform_cv_candidate_labels(args, ori_labels)
    elif args.exp_type == 'ins':
        raise NotImplementedError("You have chosen an unsupported experiment type. Please check and try again.")
        
    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1
    
    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('Partial labels correctly loaded !')
    else:
        print('Inconsistent permutation !')

    print('Average candidate num: ', partialY_matrix.sum(1).mean())
    
    partial_training_dataset = miniImagenet_Partialize(ori_data, partialY_matrix.float(), ori_labels.float())
    
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


class miniImagenet_Partialize(Dataset):
    def __init__(self, images, given_partial_label_matrix, true_labels):
        
        self.ori_images = images
        self.given_partial_label_matrix = given_partial_label_matrix
        self.true_labels = true_labels
        
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=32),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        
        each_image1 = self.transform1(self.ori_images[index])
        each_image2 = self.transform2(self.ori_images[index])
        each_label = self.given_partial_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image1, each_image2, each_label, each_true_label, index



class miniImagenet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.test_dir = os.path.join(self.root_dir, "test")

        if (self.Train):
            self.train_transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.Resize(84)
            ])
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_test()
        
        
        self._make_dataset(Train = self.Train)
        

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".jpg"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    
    def _create_class_idx_dict_test(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.test_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.test_dir) if os.path.isdir(os.path.join(self.test_dir, d))]
        
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.test_dir):
            for f in files:
                if f.endswith(".jpg"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

        
    def _make_dataset(self, Train=True):

        self.data = []
        self.targets = []
        
            
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.test_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        
        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            
            if not os.path.isdir(dirs):
                continue
            
            for root, _, files in sorted(os.walk(dirs)):
                
                for fname in sorted(files):
                    if (fname.endswith(".jpg")):
                        path = os.path.join(root, fname)
                        
                        if Train:
                            class_index = self.class_to_tgt_idx[tgt]
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            class_index = self.class_to_tgt_idx[tgt]
                            item = (path, self.class_to_tgt_idx[tgt])
                       
                        # add
                        sample = Image.open(path)
                        sample = sample.convert('RGB')
                        
                        if Train:
                            sample = self.train_transform(sample)

                        self.data.append(sample)
                        self.targets.append(class_index)
                        
                
    def __len__(self):
        return self.len_dataset

    
    def __getitem__(self, idx):
        sample, tgt = self.data[idx], self.targets[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt
