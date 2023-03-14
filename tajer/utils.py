import os
import time
import math
import yaml
import torch
import einops
import datetime

import numpy as np
from PIL import Image
import torchvision.transforms as T


def exists(x):
    return x is not None


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


# size of pytorch tensors

def get_tensor_size(tensor: torch.Tensor, return_bytes=False):
    size = tensor.element_size() * tensor.nelement()
    if return_bytes:
        return size
    return convert_size(size)


# time stuff

def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def timing():
    return datetime.datetime.now().strftime(f'%H:%M:%S.%f')[:-3]


# pytorch model parameter count

def count_parameters(model, return_int: bool = False):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if return_int:
        return n_params

    return f'{n_params:,}'


# namespace to access and set attributes with dot notation

class ConfigNamespace(dict):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


# overwrite config file information with arguments if arguments are set

def parse_options(parser):
    args = parser.parse_args()
    cfg = yaml.load(open(args.cfg, 'r'), Loader=yaml.Loader)

    # get data config
    data_cfg_file = args.data_cfg if exists(args.data_cfg) else cfg['data_cfg']
    data_cfg = yaml.load(open(data_cfg_file, 'r'), Loader=yaml.Loader)

    # set all params from config file
    for key, value in cfg.items():
        setattr(args, key, value)

    # overwrite with args
    overwrite_args = parser.parse_args()
    for key, value in vars(overwrite_args).items():
        if exists(value):
            setattr(args, key, value)

    # add data config as namespace object
    args.data = ConfigNamespace()
    for key, value in data_cfg.items():
        args.data[key] = value

    return args


# PyTorch average meter to accumulate and aggregate values

class AverageMeter:
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        val = self._handle_value(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    @staticmethod
    def _handle_value(value):
        if isinstance(value, torch.Tensor):
            return value.item()
        return value


# Image and video stuff


def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    # tensor of shape (channels, frames, height, width) -> gif
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs, duration=duration, loop=loop, optimize=optimize)
    return images


def get_original_reconstruction_image(x: torch.Tensor, x_hat: torch.Tensor, n_ims: int = 8):
    """
    Returns pillow image of original and reconstruction images. Top row are originals, bottom
    row are reconstructions. Faster than creating a figure.
    Args:
        x: Original image tensor (in range -1 to 1)
        x_hat: Reconstructed image tensor
        n_ims: Number of images of that batch to be plotted
    Returns:
        ims: Numpy array in shape [h, w, 3] with top row being originals and
            bottom row being reconstructions.
    """
    bs, c, h, w = x.shape

    n_ims = n_ims if n_ims <= bs else bs

    x = x[:n_ims, ...].detach().cpu().numpy()
    x_hat = x_hat[:n_ims, ...].detach().cpu().numpy()

    x = np.transpose(x, (0, 2, 3, 1))
    x_hat = np.transpose(x_hat, (0, 2, 3, 1))

    x = einops.rearrange(x, 'b h w c -> h (b w) c')
    x_hat = einops.rearrange(x_hat, 'b h w c -> h (b w) c')

    ims = np.concatenate((x, x_hat), axis=0)

    ims = (ims * 127.5 + 127.5).astype(np.uint8)

    return ims


if __name__ == "__main__":
    print(f"{'timing()':<18}: {timing()}")
    print(f"{'timer(s, e)':<18}: {timer(time.time(), time.time() + 900)}")
    
    print(f"{'getsize()':<18}: {get_tensor_size(torch.randn((4096, 1024)))}")
    print(f"{'count_parameters()':<18}: {count_parameters(torch.nn.Conv2d(128, 256, 5))}")
