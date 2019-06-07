# =============================================================================
# CSE 163 Final Project
# ~ Ritik Shah and Winston Chen ~
# =============================================================================

from features import get_features
from clean_data import clean
from time import time
import pandas as pd
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


DATA_FILE = 'data/freecodecamp_casual_chatroom.csv'
INTRO = (
    '\n'
    'Analyzing Message Popularity in Group Chats\n'
    '-- Ritik Shah and Winston Chen\n'
    '\n'
    "This program analyzes the freecodecamp dataset's messages\n"
    'Type the number associated with the action\n'
    '1. Run everything from scratch\n'
    "    This runs the entire program from scratch without reading pkls\n"
    '    Warning, can take a sigificant amount of time.\n'
    '2. Read from cleaned.pkl\n'
    '    cleaned.pkl is a serialized file that contains the csv data cleaned\n'
    '     into a pandas DataFrame.\n'
    '3. Read from features.pkl\n'
    '    features.pkl is a serialized file that contains the data with the\n'
    '     text processed features as columns.\n'
    '4. Run the machine learning algorithm\n'
    '    Reads in the pickle files and produces the accuracy score rating\n'
    '     the model.\n'
    '5. Run the machine learning testing suite.\n'
    "    This file was ran independently to determine the model's\n"
    '     hyperparameters.\n'
    '\n'
    'The latter numbers will run best if you have already generated the\n'
    ' pickled files and wish to rerun a certain part of the program.\n'
    'When a number is chosen, it will continue to roll through the rest it\n'
    ' without user input.\n'
    '\n'
)

INTRO_INPUT = 'Please choose a number (anything else will quit the program: '


def do_features(data):
    start = time()
    out = get_features(data)
    logger.debug(f'features took: {round(time() - start, 3)}s')
    return out


def do_clean(data):
    start = time()
    out = clean(data)
    logger.debug(f'cleaning took: {round(time() - start, 3)}s')
    return out


def sample(data, frac):
    return data.sample(frac=frac).reset_index(drop=True)


def ask_question(s):
    return str(input(s)).upper()[0] == 'Y'


def main():
    print(INTRO)
    try:
        action = int(input(INTRO_INPUT))
    except ValueError:
        print('Number not found. Quitting...')
        sys.exit()

    if ask_question('Use any stored pickles? [Y or N]: '):
        pkl = str(input('Which pickle? [Clean] or [Feature] data? ')).upper()
        logger.info('unpickling')
        if pkl == 'CLEAN':
            cleaned = pd.read_pickle('cleaned.pkl')
            cleaned = sample(cleaned, 1)
            data = do_features(cleaned)
            logger.info('saving to featured.pkl')
        elif pkl == 'FEATURE':
            data = pd.read_pickle('featured.pkl')
        else:
            print('Invalid Answer. Quitting...')
            sys.exit()
    else:
        cleaned = do_clean(DATA_FILE)
        cleaned.to_pickle('cleaned.pkl')
        cleaned = sample(cleaned, 1)
        data = do_features(cleaned)
        logger.info('saving to pickle')

    data.to_pickle('featured.pkl')  # pickle for future usage

    print(data)  # for now
    if ask_question('Debug? [Y or N]: '):
        breakpoint()  # debug


if __name__ == "__main__":
    main()
