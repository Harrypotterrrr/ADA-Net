import random

from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.datasets as dsets
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_train_loader(cfg):
    assert cfg.dataset in ['cifar10']

    if cfg.dataset == 'cifar10':
        train_data = dsets.CIFAR10(
            cfg.image_path,
            train=True,
            download=True,
            transform=transform,
        )

    index = list(range(len(train_data)))
    random.shuffle(index)
    label_index = SubsetRandomSampler(index[:cfg.label_num])
    unlabel_index = SubsetRandomSampler(index[cfg.label_num:])

    label_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sampler=label_index
    )

    unlabel_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sampler=unlabel_index
    )



    return label_loader, unlabel_loader


def get_validation_loader(cfg):
    assert cfg.dataset in ['cifar10', ]
    if cfg.dataset == 'kinetics':
        val_data = dsets.CIFAR10(
            cfg.image_path,
            train=True,
            transform=transform,
        )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    return val_loader

def get_test_loader(cfg):
    assert cfg.dataset in ['cifar10', ]
    if cfg.dataset == 'kinetics':
        test_data = dsets.CIFAR10(
            cfg.image_path,
            train=True,
            transform=transform,
        )
    test_loader = DataLoader(
        test_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    return test_loader

if __name__ == '__main__':

    import argparse
    from torchvision.utils import make_grid

    import matplotlib.pyplot as plt
    import numpy as np


    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, default='../data')
    parser.add_argument('--label_num', type=int, default=4000)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)

    config = parser.parse_args()

    label_loader, unlabel_loader = get_train_loader(config)
    dataiter = iter(label_loader)
    images, labels = dataiter.next()

    imshow(make_grid(images))