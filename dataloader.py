import random
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.datasets as dsets
import torchvision.transforms as transforms

def data_gen(loader):
    while True:
        for data in loader:
            yield data

def dataloader(dset="cifar10", path="data/cifar10", bs=100, num_workers=8, label_num=4000):
    assert dset in ["cifar10", "cifar100", "svhn"]
    
    if dset == "cifar10":
        mean = (0.49139968, 0.48215841, 0.44653091)
        std = (0.24703223, 0.24348513, 0.26158784)
    elif dset =="cifar100":
        mean = (0.50707516, 0.48654887, 0.44091784)
        std = (0.26733429, 0.25643846, 0.27615047)
    elif dset =="svhn":
        mean = (0.4376821, 0.4437697, 0.47280442)
        std = (0.19803012, 0.20101562, 0.19703614)
    normaliztion = transforms.Normalize(mean, std)
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normaliztion
            ])
    if dset == "cifar10":
        train_dataset = dsets.CIFAR10(root=path, train=True, download=True, transform=train_transform)
    elif dset == "cifar100":
        train_dataset = dsets.CIFAR100(root=path, train=True, download=True, transform=train_transform)
    elif dset == "svhn":
        train_dataset = dsets.SVHN(root=path, split='train', download=True, transform=train_transform)

    index = list(range(len(train_dataset)))
    random.shuffle(index)
    label_index = SubsetRandomSampler(index[:label_num])
    unlabel_index = SubsetRandomSampler(index[label_num:])
    label_loader = DataLoader(train_dataset, batch_size=bs, num_workers=num_workers,
                              sampler=label_index, drop_last=True)
    unlabel_loader = DataLoader(train_dataset, batch_size=bs, num_workers=num_workers,
                                sampler=unlabel_index, drop_last=True)
    
    test_transform = transforms.Compose([transforms.ToTensor(), normaliztion])
    if dset == "cifar10":
        test_dataset = dsets.CIFAR10(root=path, train=False, download=True, transform=test_transform)
    elif dset == "cifar100":
        test_dataset = dsets.CIFAR100(root=path, train=False, download=True, transform=test_transform)
    elif dset == "svhn":
        test_dataset = dsets.SVHN(root=path, split='test', download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False,
                             num_workers=num_workers)
    
    return data_gen(label_loader), data_gen(unlabel_loader), test_loader

