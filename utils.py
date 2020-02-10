import os, logging, shutil
from os.path import exists, join
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def make_folder(path):
    if not exists(path):
        os.makedirs(path)

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

clipped_cross_entropy = ClippedCrossEntropy.apply
clipped_kl_divergence = ClippedKLDivergence.apply
