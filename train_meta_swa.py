import os, argparse, torch, math, time, random
from os.path import join, isfile
import torch.nn.functional as F
from torch.optim import SGD
from torch.distributions import Beta
from tensorboardX import SummaryWriter

from dataloader import cifar10, svhn
from utils import make_folder, AverageMeter, Logger, accuracy, save_checkpoint, compute_weight, WeightSWA, update_batchnorm
from model import ConvLarge 

parser = argparse.ArgumentParser()
# Configuration
parser.add_argument('--num-label', type=int, default=4000)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'svhn'])
parser.add_argument('--aug', type=str, default=None, help='Apply ZCA augmentation')

# Training setting
parser.add_argument('--additional', type=str, default='None', choices=['None', 'label', 'unlabel'], help='Use additional data for training')
parser.add_argument('--auto-weight', action='store_true', help='Automatically adjust the weight for unlabel data')
parser.add_argument('--weight', type=float, default=1., help='re-weighting scalar for the additional loss')
parser.add_argument('--mix-up', action='store_true', help='Use mix-up augmentation')
parser.add_argument('--alpha', type=float, default=1., help='Concentration parameter of Beta distribution')
parser.add_argument('--start-step', type=int, default=0, help='Start step (for resume)')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
parser.add_argument('--epsilon', type=float, default=1e-2, help='epsilon for gradient estimation')
parser.add_argument('--lr', type=float, default=0.1, help='Maximum learning rate')
parser.add_argument('--full-interval', type=int, default=84000, help='Number of iterations in the full cosine annealing')
parser.add_argument('--cycle-interval', type=int, default=12000, help='Number of iterations in each small cosine annealing')
parser.add_argument('--num-cycles', type=int, default=33, help='Number of repeats of the small cosine annealing')
parser.add_argument('--fastswa-freq', type=int, default=1200, help='Frequency of fastSWA')
parser.add_argument('--warmup', type=int, default=4000, help='Warmup iterations')
parser.add_argument('--const-steps', type=int, default=0, help='Number of iterations of constant lr')
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

# Compute total iteration number and record start iteration
args.total_steps = args.warmup + args.const_steps + args.full_interval + args.cycle_interval * (args.num_cycles-2)
args.record_start = args.warmup + args.const_steps + args.full_interval - 2 * args.cycle_interval
# Set random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create directories if not exist
make_folder(args.save_path)
make_folder(join(args.save_path, "checkpoints"))
logger = Logger(join(args.save_path, 'log.txt'))
writer = SummaryWriter(log_dir=args.save_path)
logger.info('Called with args:')
logger.info(args)
torch.backends.cudnn.benchmark = True
# Define dataloader
logger.info("Loading data...")
if args.dataset == "cifar10": dset = cifar10
elif args.dataset == "svhn": dset = svhn
label_loader, unlabel_loader, test_loader = dset(
        args.data_path, args.batch_size, args.num_workers, args.num_label, args.aug
        )
# Build model and optimizer
logger.info("Building model and optimzer...")
model = ConvLarge(stochastic=True).cuda()
swa_model = ConvLarge(stochastic=True).cuda()
for param in swa_model.parameters(): param.detach_()
fastswa_model = ConvLarge(stochastic=True).cuda()
for param in fastswa_model.parameters(): param.detach_()

optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
swa_optim = WeightSWA(swa_model)
fastswa_optim = WeightSWA(fastswa_model)
logger.info("Model:\n%s\nOptimizer:\n%s" % (str(model), str(optimizer)))
# Optionally build beta distribution
if args.mix_up:
    beta_distribution = Beta(torch.tensor([args.alpha]), torch.tensor([args.alpha]))
# Optionally resume from a checkpoint
if args.resume is not None:
    if isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_step = checkpoint['step']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume))

def compute_lr(step):
    if step < args.warmup:
        lr = args.lr * step / args.warmup
    elif step < args.warmup + args.const_steps:
        lr = args.lr
    elif step < args.record_start:
        lr = args.lr * ( 1. + math.cos( (step-args.warmup-args.const_steps) / args.full_interval *  math.pi ) ) / 2.
    else:
        lr = args.lr * ( 1. + math.cos( ( args.full_interval - 2.*args.cycle_interval + ((step-args.record_start)%args.cycle_interval) ) / args.full_interval *  math.pi)) / 2.
    return lr

def main():
    data_times, batch_times, label_losses, unlabel_losses, label_acc, unlabel_acc = [AverageMeter() for _ in range(6)]
    best_acc, swa_best_acc, fastswa_best_acc = 0., 0., 0.
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
        _label_gt = F.one_hot(label_gt, num_classes=10).float()
        data_end = time.time()
        
        ### TODO: set inner-lr == lr for now.
        # Compute the inner learning rate and outer learning rate
        lr = inner_lr = compute_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if args.auto_weight:
            weight = compute_weight(args.weight, step, args.total_steps)
        else:
            weight = args.weight

        ### First-order Approximation ###
        _concat = lambda xs: torch.cat([x.view(-1) for x in xs])
        # Evaluation model
        model.eval()
        # Forward label data and perform backward pass
        label_pred = model(label_img)
        label_loss = F.kl_div(F.log_softmax(label_pred, dim=1), _label_gt, reduction='batchmean')
        dtheta = torch.autograd.grad(label_loss, model.parameters(), only_inputs=True)
        
        ### TODO: implement inner-iter
        with torch.no_grad():
            # Compute the unlabel pseudo-gt
            unlabel_pred = model(unlabel_img)
            unlabel_pseudo_gt = F.softmax(unlabel_pred, dim=1)
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
            unlabel_grad.div_(2.*epsilon)
            unlabel_pseudo_gt.sub_(inner_lr, unlabel_grad)
            # Label normalization
            torch.relu_(unlabel_pseudo_gt)
            sums = torch.sum(unlabel_pseudo_gt, dim=1, keepdim=True)
            unlabel_pseudo_gt /= torch.where(sums == 0., torch.ones_like(sums), sums)

        # Training mode
        model.train()
        if args.mix_up:
            # Adopt mix-up augmentation
            with torch.no_grad():
                alpha = beta_distribution.sample((args.batch_size,)).cuda()
                _alpha = alpha.view(-1, 1, 1, 1)
                interp_img = (label_img * _alpha + unlabel_img * (1. - _alpha)).detach()
                interp_pseudo_gt = (_label_gt * alpha + unlabel_pseudo_gt * (1. - alpha)).detach()
            interp_pred = model(interp_img)
            loss = unlabel_loss = F.kl_div(F.log_softmax(interp_pred, dim=1), interp_pseudo_gt, reduction='batchmean')
        else:
            # Compute loss with `unlabel_pseudo_gt`
            unlabel_pred = model(unlabel_img)
            loss = unlabel_loss = F.kl_div(F.log_softmax(unlabel_pred, dim=1), unlabel_pseudo_gt, reduction='batchmean')
        
        if args.additional == 'label':
            # Additionally use label data
            label_pred = model(label_img)
            label_loss = F.kl_div(F.log_softmax(label_pred, dim=1), _label_gt, reduction='batchmean')
            loss = label_loss + weight * loss
        elif args.additional == 'unlabel' and args.mix:
            # Additionally use unlabel data
            unlabel_pred = model(unlabel_img)
            unlabel_loss = F.kl_div(F.log_softmax(unlabel_pred, dim=1), unlabel_pseudo_gt, reduction='batchmean')
            loss = loss + weight * unlabel_loss
        
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
        label_losses.update(label_loss.item(), label_img.size(0))
        unlabel_losses.update(unlabel_loss.item(), unlabel_img.size(0))
        label_acc.update(label_top1.item(), label_img.size(0))
        unlabel_acc.update(unlabel_top1.item(), unlabel_img.size(0))
        # Print and log
        if step % args.print_freq == 0:
            logger.info("Step {0:05d} Dtime: {dtimes.avg:.3f} Btime: {btimes.avg:.3f} "
                        "Lloss: {llosses.val:.3f} (avg {llosses.avg:.3f}) Uloss: {ulosses.val:.3f} (avg {ulosses.avg:.3f}) "
                        "Lacc: {label.val:.3f} (avg {label.avg:.3f}) Uacc: {unlabel.val:.3f} (avg {unlabel.avg:.3f}) "
                        "OLR: {1:.4f} ILR: {2:.4f} W: {3:.3f}".format(
                                step, lr, inner_lr, weight,
                                dtimes=data_times, btimes=batch_times, llosses=label_losses,
                                ulosses=unlabel_losses, label=label_acc, unlabel=unlabel_acc
                                ))
        # Test and save model
        if (step + 1) % args.test_freq == 0 or step == args.total_steps - 1:
            acc = evaluate(model)
            # Write to the tfboard
            writer.add_scalar('train/label-acc', label_acc.avg, step)
            writer.add_scalar('train/unlabel-acc', unlabel_acc.avg, step)
            writer.add_scalar('train/label-loss', label_losses.avg, step)
            writer.add_scalar('train/unlabel-loss', unlabel_losses.avg, step)
            writer.add_scalar('train/lr', lr, step)
            writer.add_scalar('train/inner-lr', inner_lr, step)
            writer.add_scalar('train/weight', weight, step)
            writer.add_scalar('test/accuracy', acc, step)
            # remember best accuracy and save checkpoint
            is_best = acc > best_acc
            if is_best: best_acc = acc
            logger.info("Best Accuracy: %.5f" % best_acc)
            save_checkpoint({
                'step': step + 1,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict()
                }, is_best, path=join(args.save_path, "checkpoint"), filename="checkpoint-epoch%06d.pth"%step)
            
            if (step > args.record_start and (step - args.record_start + 1) % args.cycle_interval == 0) or step == args.total_steps - 1:
                # Evaluate and save the SWA model
                swa_optim.update(model)
                update_batchnorm(swa_model, label_loader, unlabel_loader)
                logger.info("Evaluating the SWA model:")
                swa_acc = evaluate(swa_model)
                writer.add_scalar('test/swa-accuracy', swa_acc, step)
                
                is_best = swa_acc > swa_best_acc
                if is_best: swa_beat_acc = swa_acc
                logger.info("Best SWA Accuracy: %.5f" % swa_beat_acc)
                save_checkpoint({
                    'step': step + 1,
                    'model': swa_model.state_dict(),
                    'best_acc': swa_beat_acc
                    }, is_best, path=args.save_path, filename="checkpoint-swa.pth")
            
            if (step > args.record_start and (step - args.record_start + 1) % args.fastswa_freq == 0) or step == args.total_steps - 1:
                # Evaluate and save the fastSWA model
                fastswa_optim.update(model)
                update_batchnorm(fastswa_model, label_loader, unlabel_loader)
                logger.info("Evaluating the fastSWA model:")
                fastswa_acc = evaluate(fastswa_model)
                writer.add_scalar('test/fastswa-accuracy', fastswa_acc, step)
                
                is_best = fastswa_acc > fastswa_best_acc
                if is_best: fastswa_best_acc = fastswa_acc
                logger.info("Best fastSWA Accuracy: %.5f" % fastswa_best_acc)
                save_checkpoint({
                    'step': step + 1,
                    'model': fastswa_model.state_dict(),
                    'best_acc': fastswa_best_acc
                    }, is_best, path=args.save_path, filename="checkpoint-fastswa.pth")
            # Reset the AverageMeters
            label_losses.reset()
            unlabel_losses.reset()
            label_acc.reset()
            unlabel_acc.reset()

@torch.no_grad()
def evaluate(model):
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

if __name__ == "__main__":
    # Train and evaluate the model
    main()
    writer.close()
    logger.close()
