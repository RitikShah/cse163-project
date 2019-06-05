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


def main():
    if ask_question('Used stored pickle? [Y or N]: '):
        logging.info('unpickling')
        data = pd.read_pickle('data.pkl')
    else:
        cleaned = clean(DATA_FILE)
        # data.to_pickle('data.pkl')
        start = time()
        data = get_features(cleaned)
        logging.debug(f'features took: {round(time() - start, 3)}')
        # print(data)
        logging.info('saving to pickle')
        data.to_pickle('data.pkl')  # pickle for future usage

    print(data)  # for now
    if ask_question('Debug? [Y or N]: '):
        breakpoint()  # debug


def ask_question(s):
    return str(input(s)).upper()[0] == 'Y'


if __name__ == "__main__":
    main()
