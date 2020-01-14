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
