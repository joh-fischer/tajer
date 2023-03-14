# PyTorch CSV and Tensorboard Logger

This is an implementation of a logger which includes tensorboard functionality for 
keeping track of your `PyTorch` experiments. You can easily log all metrics during training and save
them afterwards, aggregate average and sum over an epoch, and additionally wraps a `SummaryWriter`
so that you can directly log to tensorboard.

## Usage

```python
from logger import Logger
import torch

experiment_name = 'model1'

logger = Logger('./logs',
                # create log-folder: './logs/model1/22-07-07_121028'
                experiment_name, timestamp=True,
                # include tensorboard SummaryWriter
                tensorboard=True)             

logger.log_hparams({'lr': 1e-4,
                    'optimizer': 'Adam'})

for epoch in range(2):
    # initialize epoch to aggregate values
    logger.init_epoch(epoch)     

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
```

The output of the `logs/metrics.csv` file will look like this
```
    epoch  phase      loss       acc  step  val_loss
0       0  train  0.139702  0.000383     0       NaN
1       0  train  0.324003  0.049939     1       NaN
2       0  train  0.708936  0.728604     2       NaN
3       0  train  0.549540  0.665381     3       NaN
4       0    val       NaN       NaN     4  0.231040
5       0    val       NaN       NaN     5  0.806765
6       1  train  0.523456  0.199055     6       NaN
7       1  train  0.834524  0.236231     7       NaN
8       1  train  0.221323  0.863670     8       NaN
9       1  train  0.119104  0.041285     9       NaN
10      1    val       NaN       NaN    10  0.727887
11      1    val       NaN       NaN    11  0.598616
```
