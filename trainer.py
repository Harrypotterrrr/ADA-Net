import time
import datetime

import torch.nn as nn
from torch.distributions import Beta
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import save_image, make_grid
import numpy as np

from Module.arch import ResNet, Discriminator, Classifier
from utils import *


class Trainer(object):
    def __init__(self, config, label_loader=None, unlabel_loader=None):
        # Data loader
        self.label_loader = label_loader
        self.unlabel_loader = unlabel_loader

        # Configuration
        self.label_num = config.label_num
        self.total_num = config.total_num
        self.dataset = config.dataset
        self.version = config.version

        # Epoch size
        self.log_epoch = config.log_epoch
        self.sample_epoch = config.sample_epoch
        self.model_save_epoch = config.model_save_epoch

        # Training setting
        self.total_epoch = config.total_epoch
        self.lr_schr = config.lr_schr
        self.lr = config.lr
        self.lr_decay = config.lr_decay
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.lambd = config.lambd
        self.gamma = config.gamma

        # Pretrained setting
        self.pretrained_model = config.pretrained_model

        # Misc
        self.use_tensorboard = config.use_tensorboard
        self.test_batch_size = config.test_batch_size

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        # Set GPU
        self.device, self.parallel, self.gpus = set_device(config)

        # Build model
        self.epoch2step()
        self.build_model()
        self.build_opt_schr()
        self.build_loss()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            print('load_pretrained_model...')
            self.load_pretrained_model()

    def build_loss(self):

        self.beta_distr = Beta(torch.tensor([1.0]), torch.tensor([1.0]))
        self.cross_entropy = nn.CrossEntropyLoss()
        self.kl_divergence = nn.KLDivLoss(reduction='batchmean')

    def build_opt_schr(self):

        self.optimizer = torch.optim.SGD(
            [{'params': self.resnet.parameters()},
             {'params': self.classifier.parameters()},
             {'params': self.disc.parameters()}],
            lr = self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        if self.lr_schr == 'multi': # TODO
            self.lr_scher = MultiStepLR(self.optimizer, [self.total_step*0.5, self.total_step*0.75], gamma=self.lr_decay)

    def epoch2step(self):

        self.epoch = 0
        self.step_per_epoch = len(self.label_loader)
        print("steps per epoch:", self.step_per_epoch)

        self.total_step = self.total_epoch * self.step_per_epoch
        self.log_step = self.log_epoch * self.step_per_epoch
        self.sample_step = self.sample_epoch * self.step_per_epoch
        self.model_save_step = self.model_save_epoch * self.step_per_epoch

    def sample_augment(self, label_img, label_gt, unlabel_img, unlabel_pesudo):
        assert label_img.shape == unlabel_img.shape
        assert label_img.shape.__len__() == 4
        assert label_gt.shape == unlabel_pesudo.shape

        bs, *shape = label_img.shape
        alpha = self.beta_distr.sample((bs,)).to(self.device) # TODO
        _alpha = alpha.view(bs, 1, 1, 1).repeat(1, *shape)
        assert _alpha.shape == label_img.shape
        inter_img = label_img * _alpha + unlabel_img * (1-_alpha)
        a = label_gt * alpha
        inter_img_tar = label_gt * alpha + unlabel_pesudo * (1-alpha)
        inter_true = torch.stack([alpha, 1-alpha], dim=1).squeeze(2).to(self.device)

        assert inter_img.shape == label_img.shape
        assert inter_img_tar.shape == label_gt.shape
        assert inter_true.shape[0] == bs

        return inter_img, inter_img_tar, inter_true

    def train(self):

        # Data iterator
        label_iter = iter(self.label_loader)
        unlabel_iter = iter(self.unlabel_loader)

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 1

        # Start time
        print("=" * 30, "\nStart training...")
        start_time = time.time()

        for step in range(start, self.total_step + 1):

            try:
                (label_img, label_gt), (unlabel_img, _) = label_iter.next(), unlabel_iter.next()
                if label_img.shape[0] != unlabel_img.shape[0]:
                    raise Exception
            except:
                label_iter = iter(self.label_loader)
                unlabel_iter = iter(self.unlabel_loader)
                (label_img, label_gt), (unlabel_img, _) = label_iter.next(), unlabel_iter.next()

                self.epoch += 1

            label_img = label_img.to(self.device)
            # label_gt = F.one_hot(label_gt).to(self.device)
            label_gt = label_gt.to(self.device)
            unlabel_img = unlabel_img.to(self.device)

            # ============= Generate real video ============== #
            label_feature = self.resnet(label_img)
            label_pred = self.classifier(label_feature)

            res_loss = self.cross_entropy(label_pred, label_gt)

            # ============= Generate real video ============== #
            self.resnet.eval() # combine TODO
            self.classifier.eval()
            unlabel_feature = self.resnet(unlabel_img)
            unlabel_img_pseudo = self.classifier(unlabel_feature)

            # ============= mix ============== #
            self.resnet.train()
            self.classifier.train()
            self.disc.train()
            inter_img, inter_img_tar, inter_img_true = self.sample_augment(label_img, F.one_hot(label_gt).float(), unlabel_img, unlabel_img_pseudo)
            inter_feature = self.resnet(inter_img)
            inter_img_pred = self.classifier(inter_feature)
            inter_img_false = self.disc(inter_feature)


            cls_loss = self.kl_divergence(inter_img_pred.log(), inter_img_tar) # IMPORTANT log is necessary for torch kl
            disc_loss = self.kl_divergence(inter_img_false.log(), inter_img_true)

            total_loss = self.lambd * res_loss + self.gamma * (cls_loss + disc_loss)
            self.reset_grad()
            total_loss.backward()
            self.optimizer.step()
            self.lr_scher.step()

            # ==================== print & save part ==================== #
            # Print out log info
            if step % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                start_time = time.time()
                log_str = "Epoch: [%d/%d], Step: [%d/%d], time: %s, total_loss: %.4f, res_loss: %.4f, cls_loss: %.4f, disc_loss: %.4f, lr: %.2e" % \
                    (self.epoch, self.total_epoch, step, self.total_step, elapsed, total_loss, res_loss.item(), cls_loss.item(), disc_loss.item(), self.lr_scher.get_lr()[0])

                if self.use_tensorboard is True:
                    write_log(self.writer, log_str, step, total_loss, res_loss, cls_loss, disc_loss)
                print(log_str)

            # Sample images
            if step % self.sample_step == 0:
                val_img, val_gt = label_iter.next()
                val_img = val_img.to(self.device)
                val_gt = val_gt.to(self.device)

                self.resnet.eval()
                self.classifier.eval()

                feature = self.resnet(val_img)
                val_pred = self.classifier(feature)
                val_pred, indices = torch.max(val_pred, dim=1)

                acc = np.mean((indices == val_gt).cpu().numpy())
                print("accuracy:", acc)

                # for i in range(self.n_class):
                #     for j in range(self.test_batch_size):
                #         if self.use_tensorboard is True:
                #             self.writer.add_image("Class_%d_No.%d/Step_%d" % (i, j, step), make_grid(denorm(fake_videos[i * self.test_batch_size + j].data)), step)
                #         else:
                #             save_image(denorm(fake_videos[i * self.test_batch_size + j].data), os.path.join(self.sample_path, "Class_%d_No.%d_Step_%d" % (i, j, step)))
                # print('Saved sample images {}_fake.png'.format(step))
                self.resnet.train()
                self.classifier.train()

            # Save model
            if step % self.model_save_step == 0:
                torch.save(self.resnet.state_dict(),
                           os.path.join(self.model_save_path, '{}_res.pth'.format(step)))
                torch.save(self.classifier.state_dict(),
                           os.path.join(self.model_save_path, '{}_cls.pth'.format(step)))
                torch.save(self.disc.state_dict(),
                           os.path.join(self.model_save_path, '{}_dis.pth'.format(step)))

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


    def build_tensorboard(self):
        from tensorboardX import SummaryWriter
        # from logger import Logger
        # self.logger = Logger(self.log_path)

        self.writer = SummaryWriter(log_dir=self.log_path)

    def load_pretrained_model(self):
        self.resnet.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.classifier.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_Ds.pth'.format(self.pretrained_model))))
        self.disc.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_Dt.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))