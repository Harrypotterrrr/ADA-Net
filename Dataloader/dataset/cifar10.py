from torchvision.datasets.cifar import CIFAR10

from PIL import Image

class Cifar10(CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__()

    def __getitem__(self, index):
        """
        Args:
            index (list): List

        Returns:
            tuple: (image, target) where target is list of the target class.
        """
        imgs, targets = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(imgs)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target