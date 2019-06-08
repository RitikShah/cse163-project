import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ask_question(s):
    return str(input(s)).upper()[0] == 'Y'


def remove_col(dset, col):
    return dset.loc[:, dset.columns != col]


def x_y(dset, col):
    return dset.loc[:, dset.columns != col], dset[col]


def unpickle(pkl):
    logger.info(f'unpickling {pkl}')
    return pd.read_pickle(pkl)


def move_up():
    os.chdir('..')
