import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def dev():
    logger.info('unpickling')
    train = pd.read_pickle('train.pkl')
    dev = pd.read_pickle('dev.pkl')
    return train.append(dev, ignore_index=True)


def test():
    logger.info('unpickling')
    train = pd.read_pickle('train.pkl')
    test = pd.read_pickle('test.pkl')
    return train.append(test, ignore_index=True)


def main():
    logger.info('unpickling')
    data = pd.read_pickle('featured.pkl')

    logger.info('splitting into sets')
    data = pd.sample(data, frac=1.0)
    rows = len(data)
    train, dev, test = np.split(data, [int(.6 * rows), int(.8 * rows)])

    logger.info('pickling')
    train.to_pickle('train.pkl')
    dev.to_pickle('dev.pkl')
    test.to_pickle('test.pkl')
