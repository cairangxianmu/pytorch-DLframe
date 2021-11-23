# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
import torch.utils.data as data
from PIL import Image

def build_loader(config):
    config.defrost()
    dataset_train = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    if torch.cuda.device_count() <= 1:
        sampler_train = None
    else:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train,  shuffle=True)

    sampler_val = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )
    return dataset_train, dataset_val, data_loader_train, data_loader_val


def build_dataset(is_train, config):
    transform = simple_transform(config)
    dataset = add_dataset(config, transform=transform, train=is_train)
    return dataset


def simple_transform(config):
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
        # transforms.Pad((2, 4)),
        # transforms.Resize((self.im_height, self.im_width)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5])
    ])
    return transform


class add_dataset(data.Dataset):
    def __init__(self, config, transform, train=True):
        """Init USPS dataset."""
        # init params
        self.train = train
        self.config = config
        self.img_root = config.DATA.DATA_PATH
        self.transform = transform
        # Num of Train = 7438, Num ot Test 1860
        self.data = self.load_samples()
        print(len(self.data))

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        refer, test, label, id = self.data[index]
        refer_img = Image.open(os.path.join(self.img_root, refer)).convert('L')
        refer_img = self.transform(refer_img)
        test = Image.open(os.path.join(self.img_root, test)).convert('L')
        test_img = self.transform(test)
        label = torch.tensor(eval(label)).float().unsqueeze(0)
        id = torch.tensor(int(id)).unsqueeze(0)
        img = torch.cat((refer_img, test_img), dim=0)
        return img, label, id

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)


    def load_samples(self):
        """Load sample images from dataset."""
        if self.train:
            suffix = '_train.txt'
        else:
            suffix = '_test.txt'
        filename = './data/data_list/' + self.config.DATA.DATASET + suffix
        data = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data.append(line.split())
        return data
