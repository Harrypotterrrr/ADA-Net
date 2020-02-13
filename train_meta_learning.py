import os, argparse, torch, time, random
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from tensorboardX import SummaryWriter

from dataloader import cifar10
from utils import make_folder, AverageMeter, Logger, accuracy, save_checkpoint, compute_lr
from utils import CrossEntropy, KLDivergence, clipped_cross_entropy, clipped_kl_divergence
from model import FullModel

parser = argparse.ArgumentParser()
# Configuration
parser.add_argument('--num-label', type=int, default=4000)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'svhn'])
# Training setting
parser.add_argument('--use-label', action='store_true', help='Directly use label data')
parser.add_argument('--unlabel-weight', type=float, default=1., help='re-weighting scalar for unlabel data')
parser.add_argument('--total-steps', type=int, default=120000, help='Total training epochs')
parser.add_argument('--start-step', type=int, default=0, help='Start step (for resume)')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
parser.add_argument('--epsilon', type=float, default=1e-2, help='epsilon for gradient estimation')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate annealing multiplier')
parser.add_argument('--milestones', type=eval, default=[60000, 90000], help='Learning rate annealing steps')
parser.add_argument('--inner-lr', type=float, default=0.1, help='Initial inner learning rate')
parser.add_argument('--inner-gamma', type=float, default=0.1, help='Inner learning rate annealing multiplier')
parser.add_argument('--inner-milestones', type=eval, default=[60000, 90000], help='Inner learning rate annealing steps')
parser.add_argument('--inner-iter', type=int, default=1, help='Number of iterations in the inner loop')
parser.add_argument('--type', default='0', type=str, choices=['0', '1', '2', '3'], help='normalization type of updated labels')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
parser.add_argument('--resume', type=str, default=None, help='Resume model from a checkpoint')
parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
# Misc
parser.add_argument('--print-freq', type=int, default=1, help='Print and log frequency')
parser.add_argument('--test-freq', type=int, default=400, help='Test frequency')
# Path
parser.add_argument('--data-path', type=str, default='./data', help='Data path')
parser.add_argument('--save-path', type=str, default='./results/tmp', help='Save path')
args = parser.parse_args()

# Set random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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
# Build model and optimizer
logger.info("Building model and optimzer...")
model = FullModel(stochastic=True).cuda()
optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
logger.info("Model:\n%s\nOptimizer:\n%s" % (str(model), str(optimizer)))
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
    data_times, batch_times, unlabel_losses, label_acc, unlabel_acc = [AverageMeter() for _ in range(5)]
    inner_record = [AverageMeter() for _ in range(args.inner_iter)]
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
        
        # Compute the inner learning rate and outer learning rate
        inner_lr = compute_lr(args.inner_lr, step, args.inner_gamma, args.inner_milestones)
        lr = compute_lr(args.lr, step, args.gamma, args.milestones)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        ### First-order Approximation ###
        _concat = lambda xs: torch.cat([x.view(-1) for x in xs])
        # Evaluation model
        model.eval()
        # Update pseudo label for `args.inner_iter` times
        for inner_step in range(args.inner_iter):
            if inner_step == 0:
                # Forward label data
                label_pred = model(label_img)
                label_loss = F.kl_div(F.log_softmax(label_pred, dim=1), F.one_hot(label_gt, num_classes=10).float(), reduction='batchmean')
                # label_loss = clipped_kl_divergence(label_pred, F.one_hot(label_gt, num_classes=10).float())
                with torch.no_grad():
                    # Compute gradients of label data
                    dtheta = torch.autograd.grad(label_loss, model.parameters(), only_inputs=True)
                    # Compute the unlabel pseudo-gt for the initial step of inner-loop
                    unlabel_pseudo_gt = F.softmax(model(unlabel_img), dim=1)
            else:
                # Forward unlabel data
                unlabel_pred = model(label_img)
                unlabel_loss = F.kl_div(F.log_softmax(unlabel_pred, dim=1), unlabel_pseudo_gt, reduction='batchmean')
                with torch.no_grad():
                    # Compute gradients of unlabel data
                    dtheta0 = torch.autograd.grad(unlabel_loss, model.parameters(), only_inputs=True)
                    # Update model parameters
                    for p, g in zip(model.parameters(), dtheta0):
                        p.data.sub_(lr, g)
                # Forward label data
                label_pred = model(label_img)
                label_loss = F.kl_div(F.log_softmax(label_pred, dim=1), F.one_hot(label_gt, num_classes=10).float(), reduction='batchmean')
                with torch.no_grad():
                    # Compute gradients of label data at updated parameters
                    dtheta = torch.autograd.grad(label_loss, model.parameters(), only_inputs=True)
                    # Resume original params
                    for p, g in zip(model.parameters(), dtheta0):
                        p.data.add_(lr, g)
            
            with torch.no_grad():
                # Record label loss for each inner-step
                inner_record[inner_step].update(label_loss.item())
                # Compute step size for first-order approximation
                epsilon = args.epsilon / torch.norm(_concat(dtheta))
                # Forward finite difference
                for p, g in zip(model.parameters(), dtheta):
                    p.data.add_(epsilon, g)            
                unlabel_pred_pos = model(unlabel_img)
                # Backward finite difference
                for p, g in zip(model.parameters(), dtheta):
                    p.data.sub_(2.*epsilon, g)
                unlabel_pred_neg = model(unlabel_img)
                # Resume original params
                for p, g in zip(model.parameters(), dtheta):
                    p.data.add_(epsilon, g)
                # Compute (approximated) gradients w.r.t pseudo-gt of unlabel data
                unlabel_grad = F.log_softmax(unlabel_pred_pos, dim=1) - F.log_softmax(unlabel_pred_neg, dim=1)
                if args.type == '0':
                    # As before
                    unlabel_grad.div_(2.*epsilon)
                    unlabel_pseudo_gt.sub_(inner_lr, unlabel_grad)
                    torch.relu_(unlabel_pseudo_gt)
                    sums = torch.sum(unlabel_pseudo_gt, dim=1, keepdim=True)
                    unlabel_pseudo_gt /= torch.where(sums == 0., torch.ones_like(sums), sums)
                elif args.type == '1':
                    # Gradient normaliztion
                    unlabel_grad /= torch.norm(unlabel_grad, p=2, dim=1, keepdim=True)
                    unlabel_pseudo_gt.sub_(inner_lr, unlabel_grad)
                    torch.relu_(unlabel_pseudo_gt)
                    sums = torch.sum(unlabel_pseudo_gt, dim=1, keepdim=True)
                    unlabel_pseudo_gt /= torch.where(sums == 0., torch.ones_like(sums), sums)
                elif args.type == '2':
                    # Gradient clip
                    unlabel_grad.div_(2.*epsilon)
                    unlabel_grad -= torch.mean(unlabel_grad, dim=1, keepdim=True)
                    inner_lrs = torch.clamp(torch.min(unlabel_pseudo_gt / torch.clamp(unlabel_grad, min=1e-8), dim=1, keepdim=True)[0], max=inner_lr)
                    unlabel_pseudo_gt.sub_(inner_lrs * unlabel_grad)
                elif args.type == '3':
                    # Gradient normaliztion and clip
                    unlabel_grad -= torch.mean(unlabel_grad, dim=1, keepdim=True)
                    unlabel_grad /= torch.norm(unlabel_grad, p=2, dim=1, keepdim=True)
                    inner_lrs = torch.clamp(torch.min(unlabel_pseudo_gt / torch.clamp(unlabel_grad, min=1e-8), dim=1, keepdim=True)[0], max=inner_lr)
                    unlabel_pseudo_gt.sub_(inner_lrs * unlabel_grad)
                # Make sure the updated pseudo label is a valid distribution
                assert torch.all(unlabel_pseudo_gt >= -1e-4)
                assert torch.all(torch.abs(unlabel_pseudo_gt.norm(p=1, dim=1) - 1.) < 1e-4)
                torch.relu_(unlabel_pseudo_gt)
                F.normalize(unlabel_pseudo_gt, p=1, dim=1, out=unlabel_pseudo_gt)

        # Training mode
        model.train()
        # Compute loss with `unlabel_pseudo_gt`
        unlabel_pred = model(unlabel_img)
        unlabel_loss = F.kl_div(F.log_softmax(unlabel_pred, dim=1), unlabel_pseudo_gt, reduction='batchmean')
        # unlabel_loss = clipped_kl_divergence(unlabel_pred, unlabel_pseudo_gt)
        
        # Use label data
        if args.use_label:
            label_pred = model(label_img) 
            label_loss = F.kl_div(F.log_softmax(label_pred, dim=1), F.one_hot(label_gt, num_classes=10).float(), reduction='batchmean')
            loss = label_loss + args.unlabel_weight * unlabel_loss
        else:
            loss = unlabel_loss
        
        # One SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Compute accuracy
        label_top1, = accuracy(label_pred, label_gt, topk=(1,))
        unlabel_top1, = accuracy(unlabel_pred, unlabel_gt, topk=(1,))
        # Update AverageMeter stats
        data_times.update(data_end - data_start)
        batch_times.update(time.time() - data_end)
        unlabel_losses.update(unlabel_loss.item(), unlabel_img.size(0))
        label_acc.update(label_top1.item(), label_img.size(0))
        unlabel_acc.update(unlabel_top1.item(), unlabel_img.size(0))
        # Print and log
        if step % args.print_freq == 0:
            info = "Step {0:05d} Dtime: {dtimes.avg:.3f} Btime: {btimes.avg:.3f} Llosses: |".format(step, dtimes=data_times, btimes=batch_times)
            for i in range(args.inner_iter):
                info += "{llosses.val:.3f}|".format(i, llosses=inner_record[i])
            info += " Uloss: {ulosses.val:.3f} (avg {ulosses.avg:.3f}) Lacc: {label.val:.3f} (avg {label.avg:.3f}) ".format(ulosses=unlabel_losses, label=label_acc)
            info += "Uacc: {unlabel.val:.3f} (avg {unlabel.avg:.3f}) OLR: {0:.4f} ILR: {1:.4f}".format(lr, inner_lr, unlabel=unlabel_acc)
            logger.info(info)
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
            for i in range(args.inner_iter):
                writer.add_scalar('train/label-loss%d'%i, inner_record[i].avg, step)
            writer.add_scalar('train/unlabel-loss', unlabel_losses.avg, step)
            writer.add_scalar('train/lr', lr, step)
            writer.add_scalar('train/inner-lr', inner_lr, step)
            writer.add_scalar('test/accuracy', acc, step)
            # Reset the AverageMeters
            unlabel_losses.reset()
            label_acc.reset()
            unlabel_acc.reset()
            for meter in inner_record: meter.reset()

@torch.no_grad()
def evaluate():
    batch_time, losses, acc = [AverageMeter() for _ in range(3)]
    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (data, target) in enumerate(test_loader):
        # Load data
        data = data.cuda()
        target = target.cuda()
        # Compute output
        pred = model(data)
        loss = F.cross_entropy(pred, target, reduction='mean')
        # Measure accuracy and record loss
        top1, = accuracy(pred, target, topk=(1,))
        losses.update(loss.item(), data.size(0))
        acc.update(top1.item(), data.size(0))
        # Measure elapsed time
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
