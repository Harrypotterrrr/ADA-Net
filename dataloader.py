import random
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from utils import compute_zca_stats, zca_whitening

def data_gen(loader):
    while True:
        for data in loader:
            yield data

def cifar10(path='data/cifar10', bs=100, num_workers=8, label_num=4000, zca=False):
    train_dataset = dsets.CIFAR10(root=path, train=True, download=True, transform=None)
    if zca:
        mean, components = compute_zca_stats(train_dataset.data)
        normaliztion = transforms.Lambda(lambda x: zca_whitening(x, mean, components))
    else:
        normaliztion = transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    train_dataset.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normaliztion
            ])

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
            normaliztion
            ])
    test_dataset = dsets.CIFAR10(root=path, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False,
                             num_workers=num_workers)
    
    return data_gen(label_loader), data_gen(unlabel_loader), test_loader

def svhn(path='data/cifar10', bs=100, num_workers=8, label_num=4000, zca=None):
    train_dataset = dsets.SVHN(root=path, split='train', download=True, transform=None)
    if zca:
        mean, components = compute_zca_stats(train_dataset.data)
        normaliztion = transforms.Lambda(lambda x: zca_whitening(x, mean, components))
    else:
        normaliztion = transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    train_dataset.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normaliztion
            ])

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
            normaliztion
            ])
    test_dataset = dsets.SVHN(root=path, split='test', download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False,
                             num_workers=num_workers)

    return data_gen(label_loader), data_gen(unlabel_loader), test_loader
