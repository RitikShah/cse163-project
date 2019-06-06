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


def ask_question(s):
    return str(input(s)).upper()[0] == 'Y'


def main():
    if ask_question('Use any stored pickles? [Y or N]: '):
        pkl = str(input('Which pickle? [Clean] or [Feature] data? ')).upper()
        logging.info('unpickling')
        if pkl == 'Clean':
            data = pd.read_pickle('cleaned.pkl')
            data = do_features(data)
            logging.info('saving to featured.pkl')
            data.to_pickle('featured.pkl')  # pickle for future usage
        elif pkl == 'Feature':
            data = pd.read_pickle('featured.pkl')
    else:
        data = do_clean(DATA_FILE)
        data = do_features(data)
        logging.info('saving to pickle')
        data.to_pickle('featured.pkl')  # pickle for future usage

    print(data)  # for now
    if ask_question('Debug? [Y or N]: '):
        breakpoint()  # debug


if __name__ == "__main__":
    main()
