import random
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.datasets as dsets
import torchvision.transforms as transforms

def data_gen(loader):
    while True:
        for data in loader:
            yield data

def cifar10(path='data/cifar10', bs=100, num_workers=8, label_num=4000):
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])    
    train_dataset = dsets.CIFAR10(root=path, train=True, download=True,
                                  transform=train_transform)
    index = list(range(len(train_dataset)))
    random.shuffle(index)
    label_index = SubsetRandomSampler(index[:label_num])
    unlabel_index = SubsetRandomSampler(index[label_num:])
    label_loader = DataLoader(train_dataset, batch_size=bs, num_workers=num_workers,
                              sampler=label_index, drop_last=True)
    unlabel_loader = DataLoader(train_dataset, batch_size=bs, num_workers=num_workers,
                                sampler=unlabel_index, drop_last=True)

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    test_dataset = dsets.CIFAR10(root=path, train=False, download=True,
                                 transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False,
                             num_workers=num_workers)
    
    return data_gen(label_loader), data_gen(unlabel_loader), test_loader


#transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#])
#
#def get_train_loader(cfg):
#    assert cfg.dataset in ['cifar10']
#
#    if cfg.dataset == 'cifar10':
#        train_data = dsets.CIFAR10(
#            cfg.image_path,
#            train=True,
#            download=True,
#            transform=transform,
#        )
#
#    index = list(range(len(train_data)))
#    random.shuffle(index)
#    label_index = SubsetRandomSampler(index[:cfg.label_num])
#    unlabel_index = SubsetRandomSampler(index[cfg.label_num:])
#
#    label_loader = DataLoader(
#        train_data,
#        batch_size=cfg.batch_size,
#        num_workers=cfg.num_workers,
#        sampler=label_index
#    )
#
#    unlabel_loader = DataLoader(
#        train_data,
#        batch_size=cfg.batch_size,
#        num_workers=cfg.num_workers,
#        sampler=unlabel_index
#    )
#
#    return label_loader, unlabel_loader
#
#
#def get_validation_loader(cfg):
#    assert cfg.dataset in ['cifar10', ]
#    if cfg.dataset == 'cifar10':
#        val_data = dsets.CIFAR10(
#            cfg.image_path,
#            train=False,
#            transform=transform,
#        )
#    val_loader = DataLoader(
#        val_data,
#        batch_size=cfg.batch_size,
#        shuffle=True,
#        num_workers=cfg.num_workers
#    )
#    return val_loader
#
#def get_test_loader(cfg):
#    assert cfg.dataset in ['cifar10', ]
#    if cfg.dataset == 'cifar10':
#        test_data = dsets.CIFAR10(
#            cfg.image_path,
#            train=False,
#            transform=transform,
#        )
#    test_loader = DataLoader(
#        test_data,
#        batch_size=cfg.batch_size,
#        shuffle=True,
#        num_workers=cfg.num_workers
#    )
#    return test_loader
#
#if __name__ == '__main__':
#
#    import argparse
#    from torchvision.utils import make_grid
#
#    import matplotlib.pyplot as plt
#    import numpy as np
#
#
#    def imshow(img):
#        img = img / 2 + 0.5  # unnormalize
#        npimg = img.numpy()
#        plt.imshow(np.transpose(npimg, (1, 2, 0)))
#        plt.show()
#
#    parser = argparse.ArgumentParser()
#
#    parser.add_argument('--image_path', type=str, default='../data')
#    parser.add_argument('--label_num', type=int, default=4000)
#    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
#    parser.add_argument('--num_workers', type=int, default=8)
#    parser.add_argument('--batch_size', type=int, default=128)
#
#    config = parser.parse_args()
#
#    label_loader, unlabel_loader = get_train_loader(config)
#    dataiter = iter(label_loader)
#    images, targets = dataiter.next()
#    print(images.shape, targets.shape)
#    imshow(make_grid(images))