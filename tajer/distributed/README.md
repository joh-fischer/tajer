# Distributed PyTorch Training


In `min_DDP.py` you can find a minimum working example of single-node, multi-gpu training with PyTorch.
All communication between processes, as well as the multi-process spawn is handled by the functions defined
in `distributed.py`.

```python
import torch
import torch.nn as nn
import distributed as dist

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
```

### Main worker
First, you need to specify a main worker. This function is executed on every GPU.

```python
def main_worker(gpu, world_size, args):
    args.gpu = gpu
    
    if args.distributed:
        dist.init_process_group(gpu, world_size)

    """ Data """
    dataset = ...       # your dataset
    sampler = dist.data_sampler(dataset, args.distributed, shuffle=False)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=(sampler is None), sampler=sampler)

    """ Model """
    model = ...         # your model

    # determine device
    if not torch.cuda.is_available():               # cpu
        device = torch.device('cpu')
    else:                                           # single or multi gpu
        device = torch.device(f'cuda:{args.gpu}')
    model.to(device)

    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])

    """ Optimizer and Loss """
    optimizer = torch.optim.AdamW(model.parameters(), 0.0001)
    criterion = nn.CrossEntropyLoss().to(device)

    """ Run Epochs """
    for epoch in range(args.epochs):
        if dist.is_primary():
            print(f"------- Epoch {epoch+1}")
        
        if args.distributed:
            sampler.set_epoch(epoch)

        # training
        train(model, loader, criterion, optimizer, device)

    # kill process group
    dist.cleanup()
```

### Training
Then you can specify the trainings loop.

```python
def train(model, loader, criterion, optimizer, device):
    model.train()

    for it, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        y_hat = model(x)
    
        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = torch.argmax(y_hat, dim=1).eq(y).sum()
        acc = correct / y.shape[0]

        # Up until now, all metrics are per gpu/process.  If
        # we want to get the metrics over all GPUs, we need to
        # communicate between processes. You can find a nice
        # visualization of communication here:
        # https://pytorch.org/tutorials/intermediate/dist_tuto.html
        
        # synchronize metrics across gpus/processes
        loss = dist.reduce(loss)
        acc = dist.reduce(acc, 'avg')

        # metrics over all gpus, printed only in the main process
        if dist.is_primary():
            print(f"Finish iteration {it}"
                  f" - acc: {acc.cpu().item():.4f} ({correct}/{n})"
                  f" - loss: {loss.cpu().item():.4f}")
```

### Main
Now we only need to start the whole procedure.

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Multi-GPU Training')
    parser.add_argument('--gpu', default=None, type=int, metavar='GPU',
                        help='Specify GPU for single GPU training. '
                             'If not specified, it runs on all CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                        help='Per GPU batch size.')
    args = parser.parse_args()

    # If multiple GPUs are available, it starts the main_worker function on every GPU.
    # Otherwise, it just starts the main_worker once, either on CPU or a single GPU.
    dist.launch(main_worker, args)
```

### Usage

Run `min_DDP.py` with the following command on a multi-gpu machine
```
CUDA_VISIBLE_DEVICES="2,3" python3 min_DDP.py
```

The machine then only uses GPU 2 and 3 for training (attention: index starts at 0).

To run the example on a single, specific GPU, just enter
```
python3 min_DDP.py --gpu 3
```

In case the training gets interrupted without freeing the port, run
```
kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')
```
to kill all `multiprocessing.spawn` related processes. 
