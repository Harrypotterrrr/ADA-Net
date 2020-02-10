import os, argparse, torch, time
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter

from dataloader import cifar10
from utils import make_folder, AverageMeter, Logger, accuracy, save_checkpoint
from model import FullModel

parser = argparse.ArgumentParser()

# Configuration
parser.add_argument('--num_label', type=int, default=4000)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'svhn'])
# Training setting
parser.add_argument('--total_steps', type=int, default=120000, help='Total training epochs')
parser.add_argument('--start_step', type=int, default=0, help='Start step (for resume)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--epsilon', type=float, default=1e-2, help='epsilon for estimation gradients')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate annealing multiplier')
parser.add_argument('--multiplier', type=float, default=1., help='args.inner_lr=args.lr*args.multipler (for the label update)')
parser.add_argument('--fix_inner', action='store_true', help='fix the inner learning rate')
parser.add_argument('--type', default='0', type=str, choices=['0', '1', '2', '3'], help='normalization type of updated labels')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--resume', type=str, default=None, help='Resume model from a checkpoint')
# Misc
parser.add_argument('--print_freq', type=int, default=1, help='Print and log frequency')
parser.add_argument('--test_freq', type=int, default=400, help='Test frequency')
# Path
parser.add_argument('--data_path', type=str, default='./data', help='Data path')
parser.add_argument('--save_path', type=str, default='./results/tmp', help='Save path')

args = parser.parse_args()

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

# Create directories if not exist
make_folder(args.save_path)
logger = Logger(os.path.join(args.save_path, 'log.txt'))
writer = SummaryWriter(log_dir=args.save_path)
logger.info('Called with args:')
logger.info(args)

torch.backends.cudnn.benchmark = True

# Define dataloader
logger.info("Loading data...")
label_loader, unlabel_loader, test_loader = cifar10(
        args.data_path, args.batch_size, args.num_workers, args.num_label
        )

# Build model
logger.info("Building models...")
model = FullModel(stochastic=False).cuda()
criterion = CrossEntropy().cuda()

# Build optimizer and lr_scheduler
logger.info("Building optimizer and lr_scheduler...")
optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
lr_scheduler = MultiStepLR(optimizer, gamma=args.gamma,
                           milestones=[args.total_steps//2, args.total_steps*3//4])
   
# Optionally resume from a checkpoint
if args.resume is not None:
    if os.path.isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_step = checkpoint['step']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume))

def main():
    data_times, batch_times, label_losses, unlabel_losses, label_acc, unlabel_acc = [AverageMeter() for _ in range(6)]
    best_acc = 0.
    logger.info("Start training...")
    for step in range(args.start_step, args.total_steps):
        # Load data and distribute to devices
        data_start = time.time()
        label_img, label_gt = next(label_loader)
        unlabel_img, unlabel_gt = next(unlabel_loader)
        
        label_img = label_img.cuda()
        label_gt = label_gt.cuda()
        unlabel_img = unlabel_img.cuda()
        unlabel_gt = unlabel_gt.cuda()
        data_end = time.time()
        
        """
        # Compute the inner learning rate and outer learning rate
        args.inner_lr = args.lr * args.multiplier if args.fix_inner \
                        else optimizer.param_groups[0]['lr'] * args.multiplier
        lr = optimizer.param_groups[0]['lr']
        """
        
        ### First-order Approximation ###
        _concat = lambda xs: torch.cat([x.view(-1) for x in xs])
        
        model.eval()
        # Forward label data and perform backward pass
        label_pred = model(label_img)
        label_loss = F.cross_entropy(label_pred, label_gt, reduction='mean')
        dtheta = torch.autograd.grad(label_loss, model.parameters(), only_inputs=True)
        
        # Compute the unlabel pseudo-gt
        unlabel_pseudo_gt = torch.zeros(unlabel_img.size(0), 10).cuda()
        unlabel_pseudo_gt.requires_grad = True
        
        # Compute step size for first-order approximation
        epsilon = args.epsilon / torch.norm(_concat(dtheta))
        
        # positive label gradients
        for p, g in zip(model.parameters(), dtheta):
            p.data.add_(epsilon, g)
        with torch.no_grad():
            unlabel_pred = model(unlabel_img)
        loss = criterion(unlabel_pred, unlabel_pseudo_gt)
        grads_pos, = torch.autograd.grad(loss, (unlabel_pseudo_gt, ), only_inputs=True)
        
        # negative label gradients
        for p, g in zip(model.parameters(), dtheta):
            p.data.sub_(2.*epsilon, g)
        with torch.no_grad():
            unlabel_pred = model(unlabel_img)
        loss = criterion(unlabel_pred, unlabel_pseudo_gt)
        grads_neg, = torch.autograd.grad(loss, (unlabel_pseudo_gt, ), only_inputs=True)
        
        with torch.no_grad():
            # Resume original params
            for p, g in zip(model.parameters(), dtheta):
                p.data.add_(epsilon, g)
                
            ### TODO: normalization approach and whether to normalize or not
            # Compute the approximated label gradients
            unlabel_grad_neg = (grads_pos - grads_neg).div(2.*epsilon)
            if args.type == '0':
                torch.relu_(unlabel_grad_neg)
                sums = torch.sum(unlabel_grad_neg, dim=1, keepdim=True)
                unlabel_grad_neg /= torch.where(sums == 0., torch.ones_like(sums), sums)
            elif args.type == '1':
                torch.relu_(unlabel_grad_neg)
            elif args.type == '2':
                pass
        
        ### TODO: try the commented scripts -- initialize the unlabel gt as the current prediction
        """
        # Forward label data and perform backward pass
        label_pred = model(label_img)
        label_loss = F.cross_entropy(label_pred, label_gt, reduction='mean')
        dtheta = torch.autograd.grad(label_loss, model.parameters(), only_inputs=True)
        
        # Compute the unlabel pseudo-gt
        unlabel_pseudo_gt = F.softmax(model(unlabel_img), dim=1).detach()
        unlabel_pseudo_gt.requires_grad = True
        
        # Compute step size for first-order approximation
        epsilon = args.epsilon / torch.norm(_concat(dtheta))
        
        # positive label gradients
        for p, g in zip(model.parameters(), dtheta):
            p.data.add_(epsilon, g)
        unlabel_pred = model(unlabel_img)
        loss = criterion(unlabel_pred, unlabel_pseudo_gt)
        grads_pos, = torch.autograd.grad(loss, (unlabel_pseudo_gt, ), only_inputs=True)
        
        # negative label gradients
        for p, g in zip(model.parameters(), dtheta):
            p.data.sub_(2.*epsilon, g)
        unlabel_pred = model(unlabel_img)
        loss = criterion(unlabel_pred, unlabel_pseudo_gt)
        grads_neg, = torch.autograd.grad(loss, (unlabel_pseudo_gt, ), only_inputs=True)
        
        with torch.no_grad():
            # Resume original params
            for p, g in zip(model.parameters(), dtheta):
                p.data.add_(epsilon, g)
                
            # Compute the approximated label gradients
            unlabel_grad = -(grads_pos - grads_neg).div(2.*epsilon)
            grad_norm = torch.norm(unlabel_grad)
        
            # Update `unlabel_pseudo_gt`
            unlabel_pseudo_gt.requires_grad = False
            unlabel_pseudo_gt -= args.inner_lr * unlabel_grad
            ### TODO: try several alternatives
            if args.type == '0':
                torch.relu_(unlabel_pseudo_gt)
                unlabel_pseudo_gt /= torch.sum(unlabel_pseudo_gt, dim=1, keepdim=True)
            elif args.type == '1':
                torch.clamp(unlabel_pseudo_gt, min=0., max=1., out=unlabel_pseudo_gt)
                unlabel_pseudo_gt /= torch.sum(unlabel_pseudo_gt, dim=1, keepdim=True)
            elif args.type == '2':
                torch.relu_(unlabel_pseudo_gt)
        """
        
        model.train()
        # Compute loss with `unlabel_pseudo_gt`
        unlabel_pred = model(unlabel_img)
        unlabel_loss = criterion(unlabel_pred, unlabel_grad_neg)
        
        ### Baseline
        # label_pred = model(label_img) 
        # loss = F.cross_entropy(label_pred, label_gt, reduction='mean')
        ###
        
        # One SGD step
        optimizer.zero_grad()
        unlabel_loss.backward()
        ### TODO: try to include label data into training
        ### Uncomment this to incorporate label data into training
        # for p, g in zip(model.parameters(), dtheta):
        #     p.grad.add_(g)
        ###
        optimizer.step()
        lr_scheduler.step()
        
        ### Show batch-time
        # print(time.time() - data_end)
        # continue
        ###

        # Compute accuracy
        label_top1, = accuracy(label_pred, label_gt, topk=(1,))
        unlabel_top1, = accuracy(unlabel_pred, unlabel_gt, topk=(1,))
        
        # Update AverageMeter stats
        data_times.update(data_end - data_start)
        batch_times.update(time.time() - data_end)
        label_losses.update(label_loss.item(), label_img.size(0))
        unlabel_losses.update(unlabel_loss.item(), unlabel_img.size(0))
        label_acc.update(label_top1.item(), label_img.size(0))
        unlabel_acc.update(unlabel_top1.item(), unlabel_img.size(0))
            
        # Print and log
        if step % args.print_freq == 0:
            logger.info("Step{0:05d} Dtime: {dtimes.avg:.3f} Btime: {btimes.avg:.3f} "
                        "label-loss: {llosses.val:.3f} (avg {llosses.avg:.3f}) unlabel-loss: {ulosses.val:.3f} (avg {ulosses.avg:.3f}) "
                        "label-acc: {label.val:.3f} (avg {label.avg:.3f}) unlabel-acc: {unlabel.val:.3f} (avg {unlabel.avg:.3f}) LR: {1:.4f}".format(
                                step, optimizer.param_groups[0]['lr'],
                                dtimes=data_times, btimes=batch_times, llosses=label_losses,
                                ulosses=unlabel_losses, label=label_acc, unlabel=unlabel_acc
                                ))

        # Test and save model
        if (step + 1) % args.test_freq == 0 or step == args.total_steps - 1:
            acc = evaluate()
            # remember best accuracy and save checkpoint
            is_best = acc > best_acc
            if is_best:
                best_acc = acc
            logger.info("Best Accuracy: %.5f" % best_acc)
            save_checkpoint({
                'step': step + 1,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict()
                }, is_best, path=args.save_path, filename="checkpoint.pth")
            # Write to the tfboard
            writer.add_scalar('train/label-acc', label_acc.avg, step)
            writer.add_scalar('train/unlabel-acc', unlabel_acc.avg, step)
            writer.add_scalar('train/label-loss', label_losses.avg, step)
            writer.add_scalar('train/unlabel-loss', unlabel_losses.avg, step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
            writer.add_scalar('test/accuracy', acc, step)
            # Reset the AverageMeters
            label_losses.reset()
            unlabel_losses.reset
            label_acc.reset()
            unlabel_acc.reset()

@torch.no_grad()
def evaluate():
    batch_time, losses, acc = [AverageMeter() for _ in range(3)]
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data, target) in enumerate(test_loader):
        data = data.cuda()
        target = target.cuda()

        # compute output
        pred = model(data)
        loss = F.cross_entropy(pred, target, reduction='mean')
        
        # measure accuracy and record loss
        top1, = accuracy(pred, target, topk=(1,))
        losses.update(loss.item(), data.size(0))
        acc.update(top1.item(), data.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            logger.info('Test: [{0}/{1}] Time {btime.val:.3f} (avg={btime.avg:.3f}) '
                        'Test Loss {loss.val:.3f} (avg={loss.avg:.3f}) '
                        'Acc {acc.val:.3f} (avg={acc.avg:.3f})' \
                        .format(i, len(test_loader), btime=batch_time, loss=losses, acc=acc))

    logger.info(' * Accuracy {acc.avg:.5f}'.format(acc=acc))
    
    return acc.avg

# Train and evaluate the model
main()
writer.close()
logger.close()
