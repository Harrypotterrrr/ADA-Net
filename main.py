from trainer import Trainer

import os
import torch

from utils import *
from parameter import get_parameters
from Dataloader.data_loader import get_train_loader

def main(config):

    config.video_path = os.path.join(config.root_path, config.video_path)
    config.annotation_path = os.path.join(config.root_path, config.annotation_path)


    if config.train:

        print("="*30,"\nLoading data...")
        label_loader, unlabel_loader = get_train_loader(config)

    else:
        pass

    print('number class:', config.n_class)

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.log_path, config.version)

    if config.train:
        trainer = Trainer(label_loader, unlabel_loader, config)
        trainer.train()
    else:
        tester = Tester(test_loader, config)
        tester.test()


if __name__ == '__main__':
    config = get_parameters()

    for key in config.__dict__.keys():
        print(key, "=", config.__dict__[key])

    main(config)