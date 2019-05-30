# =============================================================================
# CSE 163 Final Project
# ~ Ritik Shah and Winston Chen ~
# =============================================================================

from clean_data import clean
from features import get_features
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)


DATA_FILE = 'data/freecodecamp_casual_chatroom.csv'


def main():
    if str(input('Used stored pickle? [Y or N]: ')).upper()[0] == 'Y':
        logging.info('unpickling')
        data = pd.read_pickle('data.pkl')
    else:
        data = clean(DATA_FILE)
        # data.to_pickle('data.pkl')
        data = get_features(data)
        # print(data)
        logging.info('saving to pickle')
        data.to_pickle('data.pkl')  # pickle for future usage

    print(data)  # for now
    breakpoint()  # debug


if __name__ == "__main__":
    main()
