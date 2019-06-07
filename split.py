import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def train():
    logger.info('unpickling train')
    return pd.read_pickle('pickle/5m_pkls/train.pkl')


def dev():
    logger.info('unpickling dev')
    return pd.read_pickle('pickle/5m_pkls/dev.pkl')


def test():
    logger.info('unpickling test')
    return pd.read_pickle('pickle/5m_pkls/test.pkl')


def main():
    logger.info('unpickling')
    data = pd.read_pickle('pickle/5m_pkls/featured.pkl')

    logger.info('splitting into sets')
    data = data.sample(frac=1.0).reset_index(drop=True)
    rows = len(data)
    train, dev, test = np.split(data, [int(.6 * rows), int(.8 * rows)])

    logger.info('pickling')
    train.to_pickle('pickle/5m_pkls/train.pkl')
    dev.to_pickle('pickle/5m_pkls/dev.pkl')
    test.to_pickle('pickle/5m_pkls/test.pkl')


if __name__ == '__main__':
    main()
