# MIT License Copyright (c) 2022 joh-fischer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import torch

import torch.multiprocessing as mp
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler


# launch
def launch(worker_fn, args):
    world_size = torch.cuda.device_count()

    args.single_gpu = world_size > 0 and args.gpu is not None
    args.distributed = world_size > 1 and not args.single_gpu

    if args.distributed:
        if args.distributed and "CUDA_VISIBLE_DEVICES" not in os.environ:
            raise ValueError("GPUs not specified. Please set CUDA_VISIBLE_DEVICES before"
                             " running the script or specify a single GPU via '--gpu'.")

        os.environ["NCCL_P2P_DISABLE"] = "1"
        mp.spawn(worker_fn, args=(world_size, args),
                 nprocs=world_size, join=True)
    elif args.single_gpu:
        worker_fn(args.gpu, world_size, args)
    else:
        worker_fn(0, world_size, args)


# distributed trainings functions
def init_process_group(rank, world_size, port='12352', backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    dist.init_process_group(backend, init_method="env://",
                            rank=rank, world_size=world_size)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def cleanup():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()


def is_primary():
    return get_rank() == 0


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


# data loading stuff
def data_sampler(dataset, distributed, shuffle):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)

    return None


# synchronization functions
def all_reduce(tensor, op='sum'):
    world_size = get_world_size()

    if world_size == 1:
        return tensor

    if op == 'sum':
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == 'avg':
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()
    else:
        raise ValueError(f'"{op}" is an invalid reduce operation!')

    return tensor


def reduce(tensor, op=dist.ReduceOp.SUM):
    world_size = get_world_size()

    if world_size == 1:
        return tensor

    dist.reduce(tensor, dst=0, op=op)

    return tensor


def gather(data):
    world_size = get_world_size()

    if world_size == 1:
        return [data]

    output_list = [torch.zeros_like(data) for _ in range(world_size)]

    if is_primary():
        dist.gather(data, gather_list=output_list)
    else:
        dist.gather(data)

    return output_list


def synchronize():
    world_size = get_world_size()

    if world_size == 1:
        return

    dist.barrier()
