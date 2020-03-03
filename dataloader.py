import os, random, pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.datasets as dsets
import torchvision.transforms as transforms


def data_gen(loader):
    while True:
        for data in loader:
            yield data

def dataloader(dset="cifar10", path="data", bs=100, num_workers=8, label_num=4000, extra='None'):
    assert dset in ["cifar10", "cifar100", "svhn"]
    assert extra in ['None', 'tiny_images_237k', 'tiny_images_500k']
    
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
    if dset == "svhn":
        train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normaliztion
                ])
    else:
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

class TinyImages(Dataset):
    """ Tiny Images Dataset """
    def __init__(self, which, transform=None, pkl_path='data-local/tiny_index.pkl', meta_path='data-local/cifar100_meta.meta', NO_LABEL=-1):
        assert which in ['237k', '500k'], 'Invalid options'
        with open(pkl_path, 'rb') as f:
            tinyimg_index = pickle.load(f)
        if which == '237k':
            print("Using all classes common with CIFAR-100.")
            with open(meta_path, 'rb') as f:
                cifar_labels = pickle.load(f)['fine_label_names']
            cifar_to_tinyimg = { 'maple_tree': 'maple', 'aquarium_fish' : 'fish' }
            cifar_labels = [l if l not in cifar_to_tinyimg else cifar_to_tinyimg[l] for l in cifar_labels]
            load_indices = sum([list(range(*tinyimg_index[label])) for label in cifar_labels], [])
        elif which == '500k':
            print("Using {} random images.".format(which))
            idxs_filename = os.path.join("/".join(meta_path.split("/")[:-1]), 'tiny_idxs500k.pkl')
            if os.path.isfile(idxs_filename):
                print("Loading indices from file", idxs_filename)
                load_indices = pickle.load(open(idxs_filename, 'rb'))
            else:
                print("Saving indices to file", idxs_filename)
                num_all_images = max(e for s, e in tinyimg_index.values())
                load_indices = np.arange(num_all_images)
                np.random.shuffle(load_indices)
                load_indices = load_indices[:500000] # note: we need to fix this 
                load_indices.sort()
                pickle.dump(load_indices, open(idxs_filename, 'wb'))
                print("Saved")
        # now we have load_indices
        self.indices = load_indices
        self.len = len(self.indices)
        self.transform = transform
        self.no_label = NO_LABEL
        print("Length of the dataset = {}".format(self.len))

    def __len__(self):
        return self.len

    def __getitem__(self, idx, verbose=False):
        if verbose: print("tiny to idx = {} actual idx = {}".format(idx, self.indices[idx]))
        img = self.load_tiny_image(self.indices[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.no_label

    def load_tiny_image(self, idx, data_path='data-local/images/cifar/cifar_tiny_images/tiny_images.bin'):
        img = None
        with open(data_path, 'rb') as f:
            f.seek(3072 * idx)
            img = np.fromfile(f, dtype='uint8', count=3072).reshape(3, 32, 32).transpose((0, 2, 1))
            img = Image.fromarray(np.rollaxis(img, 0, 3))
        return img