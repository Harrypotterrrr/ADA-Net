import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Configuration
    parser.add_argument('--label_num', type=int, default=4000)
    parser.add_argument('--total_num', type=int, default=50000)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'svhn'])
    parser.add_argument('--version', type=str, default='')

    # Training setting
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('-g', '--gpus', default=[], nargs='+', type=str, help='Specify GPU ids.')

    parser.add_argument('--total_epoch', type=int, default=300, help='how many times to update the generator')
    parser.add_argument('--log_epoch', type=int, default=1)
    parser.add_argument('--sample_epoch', type=int, default=1)
    parser.add_argument('--model_save_epoch', type=int, default=50)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_schr', type=str, default='multi', choices=['const', 'step', 'exp', 'multi', 'reduce'])
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lambd', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=1)

    # Pretrained setting
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--test_batch_size', type=int, default=8, help='how large batchsize for test')
    parser.add_argument('--num_workers', type=int, default=8)

    # Path
    parser.add_argument('--image_path', type=str, default='./data')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')


    config = parser.parse_args()

    return config
