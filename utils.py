import os, logging, shutil, math
from os.path import exists, join
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def make_folder(path):
    if not exists(path):
        os.makedirs(path)

def compute_lr(lr, decay_type, curr_step, total_steps, gamma, milestones):
    if decay_type == 'step':
        for milestone in milestones:
            if curr_step >= milestone:
                lr *= gamma
    elif decay_type == 'linear' and curr_step >= milestones[0]:
        lr *= 1. - (curr_step - milestones[0]) / (total_steps - milestones[0])
    elif decay_type == 'cosine' and curr_step >= milestones[0]:
        lr *= (1. + math.cos((curr_step - milestones[0]) / (total_steps - milestones[0]) *  math.pi)) / 2.
    return lr

def compute_weight(weight, step, total_steps):
    return weight * (1. - math.cos(step / total_steps * math.pi)) / 2.

def ZCA(X):

    import numpy as np

    X = X.reshape((-1, np.product(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    U, Lambda, _ = np.linalg.svd(Sigma)
    W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))

    X_ZCA = np.dot(X_centered, W.T)
    X_ZCA_rescaled = (X_ZCA - X_ZCA.min()) / (X_ZCA.max() - X_ZCA.min())

    return X_ZCA_rescaled

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

class CrossEntropy(nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropy, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        assert reduction in ['none', 'sum', 'mean'], "args: `reduction` should be 'none' or 'sum' or 'mean'."
        self.reduction = reduction
    
    def forward(self, x, target):
        r"""
        Params: x      -- of Size [bs, n_classes] (logits before Softmax)
                target -- of Size [bs, n_classes] (target probabilities)
        """
        logit = -self.log_softmax(x)
        losses = torch.sum(logit * target, dim=1)
        if self.reduction == "none":
            return losses
        elif self.reduction == "sum":
            return torch.sum(losses)
        elif self.reduction == "mean":
            return torch.mean(losses)

class KLDivergence(nn.Module):
    def __init__(self, reduction='mean'):
        super(KLDivergence, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        assert reduction in ['none', 'sum', 'mean'], "args: `reduction` should be 'none' or 'sum' or 'mean'."
        self.reduction = reduction
    
    def forward(self, x, target):
        r"""
        Params: x      -- of Size [bs, n_classes] (logits before Softmax)
                target -- of Size [bs, n_classes] (target probabilities)
        """
        logit = self.log_softmax(x)
        losses = torch.sum(target * (torch.log(target) - logit), dim=1)
        if self.reduction == "none":
            return losses
        elif self.reduction == "sum":
            return torch.sum(losses)
        elif self.reduction == "mean":
            return torch.mean(losses)

class ClippedCrossEntropy(Function):
    @staticmethod
    def forward(ctx, x, target):
        ctx.save_for_backward(x, target)
        logit = F.log_softmax(x, dim=1, _stacklevel=5)
        loss = F.kl_div(logit, target, reduction='batchmean')
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            x, target = ctx.saved_variables
            x = F.softmax(x, dim=1, _stacklevel=5)
            target_over_x = target / x
        
        grad_x, grad_target = None, None
        if ctx.needs_input_grad[0]:
            grad_x = -target_over_x / x.size(0)
            grad_x -= torch.mean(grad_x, dim=1, keepdim=True)
        if ctx.needs_input_grad[1]:
            grad_target = torch.log(target_over_x) / x.size(0)
            grad_target -= torch.mean(grad_target, dim=1, keepdim=True)
        return grad_x, grad_target
               
class ClippedKLDivergence(Function):
    @staticmethod
    def forward(ctx, x, target):
        ctx.save_for_backward(x, target)
        logit = F.log_softmax(x, dim=1, _stacklevel=5)
        loss = F.kl_div(logit, target, reduction='batchmean')
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            x, target = ctx.saved_variables
            # print("before softmax")
            # print(x)
            x = F.softmax(x, dim=1, _stacklevel=5)
            target_over_x = target / (x + 1e-8)
            # print("after softmax")
            # print(x)
        
        grad_x, grad_target = None, None
        if ctx.needs_input_grad[0]:
            grad_x = -target_over_x / x.size(0)
            # print("x")
            # print(grad_x)
            grad_x -= torch.mean(grad_x, dim=1, keepdim=True)
        if ctx.needs_input_grad[1]:
            grad_target = torch.log(target_over_x + 1e-8) / x.size(0)
            # print("target")
            # print(grad_target)
            grad_target -= torch.mean(grad_target, dim=1, keepdim=True)
        return grad_x, grad_target

clipped_cross_entropy = ClippedCrossEntropy.apply
clipped_kl_divergence = ClippedKLDivergence.apply
