# =============================================================================
# CSE 163 Final Project
# ~ Ritik Shah and Winston Chen ~
# =============================================================================

from internal.main_text import INTRO, GROUPME, FREECODECAMP
from internal.utils import unpickle, goodbye, get_input
from internal.features import get_features
from internal.machine_learning import ml
from internal.clean_data import clean
from internal.split import split
import internal.machine_learning_test as mlt
import internal.feature_analysis as fa
import internal.groupme
import internal.tests

from time import time
import logging

# logging is used throughout to help provide imput
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

DATA_FILE = 'data/freecodecamp_casual_chatroom_anon.csv'

# pickles
CLEANED_PKL = 'pickles/cleaned.pkl'
FEATURED_PKL = 'pickles/featured.pkl'
GROUPME_PKL = 'pickles/groupme.pkl'


def groupme():
    """ Runs the groupme set """
    print(GROUPME)
    action = get_input()
    if action != '':
        goodbye()

    start = time()
    train, _, test = split(unpickle(GROUPME_PKL))
    ml(train, test, 1)
    print(f'Time Elapsed: {round(time() - start, 3)}s')
    goodbye()


def freecodecamp():
    """ Handles options surrounding the primary dataset """
    print(FREECODECAMP)
    action = get_input()
    try:
        action = int(action)
    except ValueError:
        goodbye()

    if action not in ACTIONS:
        goodbye()

    start = time()

    ACTIONS[action]()

    print(f'Time Elapsed: {round(time() - start, 3)}s')


def scratch(percent):
    """ Handles running the program from scratch """
    data = clean(DATA_FILE, percent)
    data = data.sample(frac=1.0).reset_index(drop=True)

    logger.info(f'saving to pickle: {CLEANED_PKL}')
    data.to_pickle(CLEANED_PKL)

    data = get_features(data)
    logger.info(f'saving to pickle: {FEATURED_PKL}')
    data.to_pickle(FEATURED_PKL)

    train, _, test = split(data)
    ml(train, test, 10)


def percent100():
    """ Runs scratch at 100% of database """
    scratch(1.00)


def percent10():
    """ Runs scratch at 10% of database """
    scratch(0.10)


def percent1():
    """ Runs scratch at 1% of database """
    scratch(0.01)


def from_pickle():
    """ Runs the machine learning model with the featured.pkl """
    data = unpickle(FEATURED_PKL)
    train, _, test = split(data)
    ml(train, test, 10)


# Dictionaries provide easy key -> function sequence
DATASETS = {
    'freecodecamp': freecodecamp,
    'groupme': groupme,
    'groupme_dev': internal.groupme.main  # hidden option used on groupme data
}

ACTIONS = {
    1: percent100,
    2: percent10,
    3: percent1,
    4: from_pickle,
    5: mlt.main,
    6: fa.main,
    7: internal.tests.main,
    8: internal.tests.playground
}


def main():
    """ Command line interface to work with entire program """
    print(INTRO)
    action = get_input()

    if action not in DATASETS:
        goodbye()

    DATASETS[action]()

    print('Press enter to quit')
    get_input()
    goodbye()


if __name__ == "__main__":
    main()
