import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets, NPY_test_datasets
from tensorboardX import SummaryWriter
from models.egeunet import EGEUNet

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config
from configs.config_setting import setting_test_config

import warnings

warnings.filterwarnings("ignore")


def validate_model(config):
    # Prepare logger
    log_dir = os.path.join(config.work_dir, 'log')
    global logger
    logger = get_logger('validate', log_dir)





    # Load dataset
    # print('#----------Preparing dataset----------#')
    # val_dataset = NPY_datasets(config.data_path, config, test=True)
    # val_loader = DataLoader(val_dataset,
    #                         batch_size=1,
    #                         shuffle=False,
    #                         pin_memory=True,
    #                         num_workers=config.num_workers,
    #                         drop_last=True)


    print('#----------Preparing test dataset----------#')
    test_dataset = NPY_test_datasets(config.data_path, config, test=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)






    # Load model
    print('#----------Preparing Model----------#')
    model_cfg = config.model_config
    if config.network == 'egeunet':
        model = EGEUNet(num_classes=model_cfg['num_classes'],
                        input_channels=model_cfg['input_channels'],
                        c_list=model_cfg['c_list'],
                        bridge=model_cfg['bridge'],
                        gt_ds=model_cfg['gt_ds'])
    else:
        raise Exception('network in not right!')
    model = model.cuda()

    # Load best weights
    print('#----------Loading best weights----------#')

    #----------path to your weights----------#
    best_weight_path = " "

    
    if not os.path.exists(best_weight_path):
        print("Error: No best weights found!")
        return

    best_weight = torch.load(best_weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(best_weight)

    # Validate on the validation set and compute accuracy
    criterion = config.criterion
    loss = test(test_loader, model, criterion, logger, config)

    print(f"Validation Loss: {loss:.4f}")

    # Compute other metrics (like accuracy) if needed


if __name__ == '__main__':
    config = setting_test_config
    validate_model(config)
