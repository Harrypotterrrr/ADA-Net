import os, logging, shutil, math, torch
from os.path import exists, join
import numpy as np

def make_folder(path):
    if not exists(path):
        os.makedirs(path)

def compute_weight(weight, step, total_steps, rampup_step=4000):
    # return weight * (1. - math.cos(step / total_steps * math.pi)) / 2.
    return weight if step > rampup_step else weight * step / rampup_step

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

class WeightSWA(object):
    def __init__(self, swa_model):
        self.num_updates = 0
        self.swa_model = swa_model # assume that the parameters are to be discarded at the first update

    def update(self, student_model):
        self.num_updates += 1
        print("Updating SWA. Current num_updates = %d"%self.num_updates)
        if self.num_updates == 1:
            self.swa_model.load_state_dict(student_model.state_dict())
        else:
            inv = 1. / float(self.num_updates)
            for swa_p, src_p in zip(self.swa_model.parameters(), student_model.parameters()):
                swa_p.data.sub_(inv*swa_p.data)
                swa_p.data.add_(inv*src_p.data)
    
    def reset(self):
        self.num_updates = 0

@torch.no_grad()
def update_batchnorm(model, label_loader, unlabel_loader, num_iters=100):
    model.train()
    for i in range(num_iters):
        label_img, _ = next(label_loader)
        unlabel_img, _ = next(unlabel_loader)
        label_img = label_img.cuda()
        unlabel_img = unlabel_img.cuda()
        model(label_img)
        model(unlabel_img)
        