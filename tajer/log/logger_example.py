import os
import torch
import pandas as pd
from logger import Logger


experiment_name = 'model1'

logger = Logger('./logs',
                # create log-folder: './logs/model1/22-07-07_121028'
                experiment_name, timestamp=True,
                # include tensorboard SummaryWriter
                tensorboard=True)

logger.log_hparams({'lr': 1e-4,
                    'optimizer': 'Adam'})

for epoch in range(2):
    logger.init_epoch(epoch)     # initialize epoch to aggregate values

    # training
    for step in range(4):
        logger.log_metrics({'loss': torch.rand(1), 'acc': torch.rand(1)},
                           phase='train', aggregate=True)

    # write to tensorboard
    logger.tensorboard.add_scalar('train/loss', logger.epoch['loss'].avg)

    # validation simulation
    for step in range(2):
        logger.log_metrics({'val_loss': torch.rand(1)},
                           phase='val', aggregate=True)

        print('Running average:', logger.epoch['val_loss'].avg)
        print('Running sum:', logger.epoch['val_loss'].sum)

logger.save()

metrics_path = os.path.join(logger.log_dir, 'metrics.csv')
print(pd.read_csv(metrics_path).head(20))
