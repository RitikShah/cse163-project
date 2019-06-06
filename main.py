# =============================================================================
# CSE 163 Final Project
# ~ Ritik Shah and Winston Chen ~
# =============================================================================

from features import get_features
from clean_data import clean
from time import time
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)


DATA_FILE = 'data/freecodecamp_casual_chatroom.csv'


def do_features(data):
    start = time()
    out = get_features(data)
    logging.debug(f'features took: {round(time() - start, 3)}')
    return out


def do_clean(data):
    start = time()
    out = clean(data)
    logging.debug(f'cleaning took: {round(time() - start, 3)}')
    return out


def sample(data, frac):
    return data.sample(frac=frac).reset_index(drop=True)


def ask_question(s):
    return str(input(s)).upper()[0] == 'Y'


def main():
    if ask_question('Use any stored pickles? [Y or N]: '):
        pkl = str(input('Which pickle? [Clean] or [Feature] data? ')).upper()
        logging.info('unpickling')
        if pkl == 'CLEAN':
            data = pd.read_pickle('cleaned.pkl')
            data = sample(data, 0.1)
            data = do_features(data)
            logging.info('saving to featured.pkl')
        elif pkl == 'FEATURE':
            data = pd.read_pickle('featured.pkl')
    else:
        data = do_clean(DATA_FILE)
        data = sample(data, 0.1)
        data = do_features(data)
        logging.info('saving to pickle')

    data.to_pickle('featured.pkl')  # pickle for future usage

    print(data)  # for now
    if ask_question('Debug? [Y or N]: '):
        breakpoint()  # debug


if __name__ == "__main__":
    main()
