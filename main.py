from Dataloader.data_loader import get_train_loader, get_test_loader
from parameter import get_parameters
from utils import *

from trainer import Trainer
from tester import Tester

def main(config):

    if config.train:
        print("="*30,"\nLoading data...")
        label_loader, unlabel_loader = get_train_loader(config)
    else:
        print("="*30,"\nLoading data...")
        test_loader = get_test_loader(config)

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.log_path, config.version)

    if config.train:
        trainer = Trainer(config, label_loader, unlabel_loader)
        trainer.train()
    else:
        tester = Tester(config, test_loader)
        tester.test()


if __name__ == '__main__':
    config = get_parameters()

    for key in config.__dict__.keys():
        print(key, "=", config.__dict__[key])

    main(config)