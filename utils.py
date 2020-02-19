import os, logging, shutil, math, torch
from os.path import exists, join
import numpy as np

def make_folder(path):
    if not exists(path):
        os.makedirs(path)

def compute_weight(weight, step, total_steps):
    return weight * (1. - math.cos(step / total_steps * math.pi)) / 2.

def zca_whitening(x, norm_stats=None):
    shape = x.shape
    x = x.reshape((-1, np.product(x.shape[1:])))
    if norm_stats is None:
        mean = np.mean(x, axis=0, keepdims=True)
        x_centered = x - mean
        sigma = np.dot(x_centered.T, x_centered) / x_centered.shape[0]
        U, Lambda, _ = np.linalg.svd(sigma)
        components = U.dot( np.diag( 1. / (np.sqrt(Lambda)+1e-5) ) ).dot(U.T)
    else: mean, components = norm_stats
    whiten = np.dot(x - mean, components.T).reshape(*shape)
    return whiten, (mean, components)

class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger("Logger")
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.setLevel(logging.INFO)
    
    def info(self, txt):
        self.logger.info(txt)
    
    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, path, filename="checkpoint.pth"):
    torch.save(state, join(path, filename))
    if is_best:
        shutil.copyfile(join(path, filename), join(path, 'model_best.pth'))

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class CosAnnealingLR(object):
    def __init__(self, loader_len, epochs, lr_max, warmup_epochs=0, last_epoch=-1):
        max_iters = loader_len * epochs
        warmup_iters = loader_len * warmup_epochs
        assert lr_max >= 0
        assert warmup_iters >= 0
        assert max_iters >= 0 and max_iters >= warmup_iters

        self.max_iters = max_iters
        self.lr_max = lr_max
        self.warmup_iters = warmup_iters
        self.last_epoch = last_epoch

        assert self.last_epoch >= -1
        self.iter_counter = (self.last_epoch+1) * loader_len
        self.lr = 0 
    
    def restart(self, lr_max=None):
        if lr_max:
            self.lr_max = lr_max
        self.iter_counter = 0 

    def step(self):
        self.iter_counter += 1
        if self.warmup_iters > 0 and self.iter_counter <= self.warmup_iters:
            self.lr = float(self.iter_counter / self.warmup_iters) * self.lr_max
        else:
            self.lr = (1 + math.cos((self.iter_counter-self.warmup_iters) / \
                                    (self.max_iters - self.warmup_iters) * math.pi)) / 2 * self.lr_max
        return self.lr
