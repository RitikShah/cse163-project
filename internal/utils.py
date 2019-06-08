import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ask_question(s):
    """ Simple method that handles input from question asking via input """
    return str(input(s)).upper()[0] == 'Y'


def remove_col(dset, col):
    """ Removes a column from a dataset """
    return dset.loc[:, dset.columns != col]


def x_y(dset, col):
    """ Returns the x and y from a dataset"""
    return dset.loc[:, dset.columns != col], dset[col]


def unpickle(pkl):
    """ Unpickles a given pkl"""
    logger.info(f'unpickling {pkl}')
    return pd.read_pickle(pkl)
