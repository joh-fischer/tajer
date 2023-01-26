import os, sys
import logging
import datetime
import functools
from termcolor import colored


@functools.lru_cache()
def get_logger(log_dir, dist_rank=0):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M%S')

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # file writer
    fh = logging.FileHandler(os.path.join(log_dir, 'log_' + timestamp + '.txt'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)

    # standard output (print), only on rank 0
    if dist_rank == 0:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(sh)

    return logger


if __name__ == "__main__":
    l = get_logger('logs')
    l.info("Information")
    l.warning("Warning")
