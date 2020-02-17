import random
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from utils import zca_whitening

def data_gen(loader):
    while True:
        for data in loader:
            yield data

def cifar10(path='data/cifar10', bs=100, num_workers=8, label_num=4000, aug=None):
    train_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ]
    if aug != 'zca':
        train_transform.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    train_transform = transforms.Compose(train_transform)
    train_dataset = dsets.CIFAR10(root=path, train=True, download=True, transform=train_transform)
    if aug == 'zca':
        train_dataset.data, norm_stats = zca_whitening(train_dataset.data)

    index = list(range(len(train_dataset)))
    random.shuffle(index)
    label_index = SubsetRandomSampler(index[:label_num])
    unlabel_index = SubsetRandomSampler(index[label_num:])
    label_loader = DataLoader(train_dataset, batch_size=bs, num_workers=num_workers,
                              sampler=label_index, drop_last=True)
    unlabel_loader = DataLoader(train_dataset, batch_size=bs, num_workers=num_workers,
                                sampler=unlabel_index, drop_last=True)

    test_transform = [transforms.ToTensor()]
    if aug != 'zca':
        test_transform.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    test_transform = transforms.Compose(test_transform)
    test_dataset = dsets.CIFAR10(root=path, train=False, download=True, transform=test_transform)

    if aug == 'zca':
        test_dataset.data, _ = zca_whitening(test_dataset.data, norm_stats)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False,
                             num_workers=num_workers)
    
    return data_gen(label_loader), data_gen(unlabel_loader), test_loader

def svhn(path='data/cifar10', bs=100, num_workers=8, label_num=4000, aug=None):
    train_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ]
    if aug != 'zca':
        train_transform.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    train_transform = transforms.Compose(train_transform)
    train_dataset = dsets.SVHN(root=path, split='train', download=True, transform=train_transform)
    if aug == 'zca':
        train_dataset.data, norm_stats = zca_whitening(train_dataset.data)

    index = list(range(len(train_dataset)))
    random.shuffle(index)
    label_index = SubsetRandomSampler(index[:label_num])
    unlabel_index = SubsetRandomSampler(index[label_num:])
    label_loader = DataLoader(train_dataset, batch_size=bs, num_workers=num_workers,
                              sampler=label_index, drop_last=True)
    unlabel_loader = DataLoader(train_dataset, batch_size=bs, num_workers=num_workers,
                                sampler=unlabel_index, drop_last=True)

    test_transform = [transforms.ToTensor()]
    if aug != 'zca':
        test_transform.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    test_transform = transforms.Compose(test_transform)
    test_dataset = dsets.SVHN(root=path, split='test', download=True, transform=test_transform)

    if aug == 'zca':
        test_dataset.data, _ = zca_whitening(test_dataset.data, norm_stats)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False,
                             num_workers=num_workers)
    
    return data_gen(label_loader), data_gen(unlabel_loader), test_loader
