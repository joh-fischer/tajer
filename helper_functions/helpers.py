import math
import torch
import datetime


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def getsize(tensor: torch.Tensor):
    size = tensor.element_size() * tensor.nelement()
    return convert_size(size)


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def timing():
    return datetime.datetime.now().strftime(f'%H:%M:%S.%f')[:-3]


def exists(variable):
    return variable is not None


def count_parameters(model: torch.nn.Module, return_int: bool = False):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if return_int:
        return n_params

    return f'{n_params:,}'


class Hyperparams(dict):
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
