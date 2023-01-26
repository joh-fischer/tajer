import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# DDP stuff
from torch.nn.parallel import DistributedDataParallel as DDP
import distributed_pytorch as dist


parser = argparse.ArgumentParser(description='PyTorch Multi-GPU Training')
parser.add_argument('--gpu', default=None, type=int, metavar='GPU',
                    help='Specify GPU for single GPU training. If not specified, it runs on all '
                         'CUDA_VISIBLE_DEVICES.')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='Number of training epochs.')
parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                    help='Batch size.')

# data
parser.add_argument('--n-classes', default=10, type=int, metavar='N',
                    help='Number of classes for fake dataset.')
parser.add_argument('--data-size', default=32, type=int, metavar='N',
                    help='Size of fake dataset.')
parser.add_argument('--image-size', default=64, type=int, metavar='N',
                    help='Size of input images.')


class DummyDataset(Dataset):
    def __init__(self, length, im_size, n_classes):
        self.len = length
        self.data = torch.randn(length, 3, im_size, im_size)
        self.labels = torch.randint(0, n_classes, (length,),
                                    generator=torch.Generator().manual_seed(0))

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return self.len


class DummyModel(nn.Module):
    def __init__(self, n_classes, in_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=7)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lin = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)

        return x


# Main workers ##################
def main_worker(gpu, world_size, args):
    args.gpu = gpu

    if args.distributed:
        dist.init_process_group(gpu, world_size)

    """ Data """
    dataset = DummyDataset(args.data_size, args.image_size, args.n_classes)
    sampler = dist.data_sampler(dataset, args.distributed, shuffle=False)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=(sampler is None), sampler=sampler)

    """ Model """
    model = DummyModel(args.n_classes)

    # determine device
    if not torch.cuda.is_available():  # cpu
        device = torch.device('cpu')
    else:  # single or multi gpu
        device = torch.device(f'cuda:{args.gpu}')
    model.to(device)

    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])

    """ Optimizer and Loss """
    # optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), 0.0001)
    criterion = nn.CrossEntropyLoss().to(device)

    """ Run Epochs """
    for epoch in range(args.epochs):
        if dist.is_primary():
            print(f"------- Epoch {epoch + 1}")

        if args.distributed:
            sampler.set_epoch(epoch)

        # training
        train(model, loader, criterion, optimizer, device)

    # kill process group
    dist.cleanup()


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
        n = y.shape[0]

        # metrics per gpu/process
        print(f"Device: {x.device}"
              f"\n\tInput: \t{x.shape}"
              f"\n\tLoss:  \t{loss.cpu().item():.5f}"
              f"\n\tAcc:   \t{correct / n:.5f} ({correct}/{n})")

        # synchronize metrics across gpus/processes
        loss = dist.reduce(loss)
        correct = dist.reduce(correct)
        n = dist.reduce(torch.tensor(n).to(device))
        acc = correct / n

        # metrics over all gpus, printed only on the main process
        if dist.is_primary():
            print(f"Finish iteration {it}"
                  f" - acc: {acc.cpu().item():.4f} ({correct}/{n})"
                  f" - loss: {loss.cpu().item():.4f}")


if __name__ == "__main__":
    # only run once
    parsed_args = parser.parse_args()
    for name, val in vars(parsed_args).items():
        print("{:<12}: {}".format(name, val))

    # start different processes, if multi-gpu is available
    # otherwise, it just starts the main_worker once
    dist.launch(main_worker, parsed_args)
