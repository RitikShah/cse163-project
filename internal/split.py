from .utils import unpickle

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURED_PKL = 'pickles/featured.pkl'

TRAIN_PKL = 'pickles/train.pkl'
DEV_PKL = 'pickles/dev.pkl'
TEST_PKL = 'pickles/test.pkl'


def get_train():
    """ unpickles training pkl and returns it """
    logger.info('unpickling train')
    return pd.read_pickle(TRAIN_PKL)


def get_dev():
    """ unpickles dev pkl and returns it """
    logger.info('unpickling dev')
    return pd.read_pickle(DEV_PKL)


def get_test():
    """ unpickles testing pkl and returns it """
    logger.info('unpickling test')
    return pd.read_pickle(TEST_PKL)


def split(data):
    """ Splits data into: train, dev, and test set as a 60:20:20 ratio """
    logger.info('splitting into train, dev, test sets')
    data = data.sample(frac=1.0).reset_index(drop=True)
    # data = data.loc[:, data.columns != 'wordCount']
    rows = len(data)
    train, dev, test = np.split(data, [int(.6 * rows), int(.8 * rows)])

    logger.info(f'pickling: {TRAIN_PKL}, {DEV_PKL}, {TEST_PKL}')
    train.to_pickle(TRAIN_PKL)
    dev.to_pickle(DEV_PKL)
    test.to_pickle(TEST_PKL)

    return train, dev, test


def main():
    """ Calls split (when split run by itself) """
    split(unpickle(FEATURED_PKL))


if __name__ == '__main__':
    main()
