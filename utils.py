import os
import torch
from torch.nn import init
import torch.nn.functional as F

def make_folder(path, version):
    if not os.path.exists(os.path.join(path, version)):
        os.makedirs(os.path.join(path, version))

def set_device(config):

    if config.gpus == "": # cpu
        return 'cpu', False, ""
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(config.gpus)

        if torch.cuda.is_available() is False: # cpu
            return 'cpu', False, ""
        else:
            # gpus = config.gpus.split(',') # if config.gpus is a list
            # gpus = (',').join(list(map(str, range(0, len(gpus))))) # generate a list of string number from 0 to len(config.gpus)
            gpus = list(range(len(config.gpus)))
            if config.parallel is True and len(gpus) > 1: # multi gpus
                return 'cuda:0', True, gpus
            else: # single gpu
                return 'cuda:'+ str(gpus[0]), False, gpus

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

def write_log(writer, log_str, step, ds_loss_real, ds_loss_fake, ds_loss, dt_loss_real, dt_loss_fake, dt_loss, g_loss):

    writer.add_scalar('data/ds_loss_real', ds_loss_real.item(), step)
    writer.add_scalar('data/ds_loss_fake', ds_loss_fake.item(), step)
    writer.add_scalar('data/ds_loss', ds_loss.item(), step)
    writer.add_scalar('data/dt_loss_real', dt_loss_real.item(), step)
    writer.add_scalar('data/dt_loss_fake', dt_loss_fake.item(), step)
    writer.add_scalar('data/dt_loss', dt_loss.item(), step)
    writer.add_scalar('data/g_loss_fake', g_loss.item(), step)

    writer.add_text('logs', log_str, step)
