import time
import datetime

import torch.nn as nn
from torch.distributions import Beta
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import save_image, make_grid

from Module.arch import ResNet, Discriminator, Classifier
from utils import *


class Tester():

    def __init__(self, config, test_loader):

        self.test_loader = test_loader

        self.version = config.version
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        # Set GPU
        self.device, self.parallel, self.gpus = set_device(config)

        # Pretrained setting
        self.pretrained_model = config.pretrained_model

        self.build_model()
        self.load_pretrained_model()

    def build_model(self):

        print("=" * 30, '\nBuild_model...')

        self.resnet = ResNet().cuda()
        self.classifier = Classifier().cuda()
        self.disc = Discriminator().cuda()

        if self.parallel:
            print('Use parallel...')
            print('gpus:', os.environ["CUDA_VISIBLE_DEVICES"])

            self.resnet = nn.DataParallel(self.resnet, device_ids=self.gpus)
            self.classifier = nn.DataParallel(self.classifier, device_ids=self.gpus)
            self.disc = nn.DataParallel(self.disc, device_ids=self.gpus)

    def load_pretrained_model(self):
        self.resnet.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_res.pth'.format(self.pretrained_model))))
        self.classifier.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_cls.pth'.format(self.pretrained_model))))
        self.disc.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_dis.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def test(self):

        test_loader = iter(self.test_loader)

        self.resnet.eval()
        self.classifier.eval()
        self.disc.eval()

        print("=" * 30, "\nStart testing...")

        ctr = torch.tensor(0).to(self.device)

        for i, (test_img, test_gt) in enumerate(test_loader):
            test_img = test_img.to(self.device)
            test_gt = test_gt.to(self.device)

            feature = self.resnet(test_img)
            pred = self.classifier(feature)
            pred, indices = torch.max(pred, dim=1)
            tag = self.disc(feature)

            ctr += torch.sum(indices == test_gt)

        print("correct prediction: ", ctr)
        print("acc: %.2f%%" % (ctr / 50000. * 100))


