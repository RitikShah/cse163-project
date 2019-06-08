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

from time import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

DATA_FILE = 'data/freecodecamp_casual_chatroom.csv'

# pickles
CLEANED_PKL = 'pickles/cleaned.pkl'
FEATURED_PKL = 'pickles/featured.pkl'
GROUPME_PKL = 'pickles/groupme.pkl'


def groupme():
    print(GROUPME)
    action = get_input()
    if action != '':
        goodbye()

    train, _, test = split(unpickle(GROUPME_PKL))
    ml(train, test, 1)
    goodbye()


def freecodecamp():
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
    scratch(1.00)


def percent10():
    scratch(0.10)


def percent1():
    scratch(0.01)


def from_pickle():
    data = unpickle(FEATURED_PKL)
    train, _, test = split(data)
    ml(train, test, 10)


DATASETS = {
    'freecodecamp': freecodecamp,
    'groupme': groupme
}

ACTIONS = {
    1: percent100,
    2: percent10,
    3: percent1,
    4: from_pickle,
    5: mlt.main,
    6: fa
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
