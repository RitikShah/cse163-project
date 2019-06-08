from pathlib import Path
import pandas as pd
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTRO_INPUT = '> '


def get_input():
    """ Grabs input as string and handles CTRL+C """
    try:
        inn = str(input(INTRO_INPUT))
    except KeyboardInterrupt:
        goodbye()

    return inn.lower().strip()


def goodbye():
    """ Prints goodbye and quits"""
    print('\nGoodbye')
    sys.exit()


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
    if not Path(pkl).exists():
        print('Pickle not found. Consider running options 1 through 3')
        goodbye()
    return pd.read_pickle(pkl)
