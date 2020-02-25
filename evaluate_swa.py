import os, argparse, torch, time, random
from os.path import join, isfile
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from dataloader import dataloader
from utils import make_folder, AverageMeter, Logger, accuracy, save_checkpoint, WeightSWA, update_batchnorm
from model import ConvLarge, shakeshake26

parser = argparse.ArgumentParser()
parser.add_argument('--start-step', type=int, default=0, help='Start step (for resume)')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
parser.add_argument('--num-label', type=int, default=4000)
parser.add_argument('--num-steps', type=int, default=100, help='Number of steps for updating BN stats')
parser.add_argument('--average-start', type=int, default=100, help='Start averaging at which epoch')
parser.add_argument('--average-interval', type=int, default=100, help='Average weight at which frequency')
parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
parser.add_argument('--resume', type=str, default=None, help='Resume model from a checkpoint')
parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
parser.add_argument('--print-freq', type=int, default=20, help='Print and log frequency')
parser.add_argument('--data-path', type=str, default='./data', help='Data path')
parser.add_argument('--save-path', type=str, default='./results/tmp', help='Save path')
args = parser.parse_args()
args.num_classes = 100 if args.dataset == 'cifar100' else 10

# Set random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create directories if not exist
make_folder(args.save_path)
logger = Logger(join(args.save_path, 'log.txt'))
writer = SummaryWriter(log_dir=args.save_path)
logger.info('Called with args:')
logger.info(args)
torch.backends.cudnn.benchmark = True
# Define dataloader
logger.info("Loading data...")
label_loader, unlabel_loader, test_loader = dataloader(
        args.dataset, args.data_path, args.batch_size, args.num_workers, args.num_label
        )
# Build model and optimizer
logger.info("Building model and optimzer...")
if args.architecture == "convlarge":
    model = ConvLarge(num_classes=args.num_classes, stochastic=True).cuda()
elif args.architecture == "shakeshake":
    model = shakeshake26(num_classes=args.num_classes).cuda()
for param in model.parameters():
    param.detach_()
swa_optim = WeightSWA(model)
logger.info("Model:\n%s" % str(model))
# Resume from a checkpoint
if args.resume is not None:
    if isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_step = checkpoint['step']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model'])
        logger.info("=> loaded checkpoint '{}' (step {})".format(args.resume, checkpoint['step']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume))

def main():
    best_acc, epoch = 0., args.average_start
    logger.info("Start evaluating SWA...")
    while True:
        filename = join(args.resume, "checkpoint-epoch%d.pth" % epoch)
        if isfile(filename):
            checkpoint = torch.load(filename)
            logger.info("=> loaded checkpoint from %s".format(filename))
        else:
            logger.info("=> No checkpoint at %s. Stop evaluation." % filename)
            break
        # Evaluate and save the SWA model
        swa_optim.update(checkpoint['model'])
        update_batchnorm(model, label_loader, unlabel_loader, num_iters=args.num_steps)
        logger.info("Evaluating the SWA model:")
        swa_acc = evaluate(model)
        writer.add_scalar('test/swa-accuracy', swa_acc, epoch)
        # Record the best evaluation result
        is_best = swa_acc > best_acc
        if is_best:
            beat_acc = swa_acc
        logger.info("Best SWA Accuracy: %.5f" % beat_acc)
        save_checkpoint({
            'model': model.state_dict(),
            'best_acc': beat_acc
            }, is_best, path=args.save_path, filename="checkpoint-swa.pth")
        epoch += args.average_interval

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
    main()
    writer.close()
    logger.close()
