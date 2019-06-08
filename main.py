# =============================================================================
# CSE 163 Final Project
# ~ Ritik Shah and Winston Chen ~
# =============================================================================

from internal.main_text import INTRO, INTRO_INPUT, GROUPME, FREECODECAMP
from internal.features import get_features
from internal.machine_learning import ml
from internal.clean_data import clean
from internal.utils import unpickle
from internal.split import split

from time import time
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')


DATA_FILE = 'data/freecodecamp_casual_chatroom.csv'

# pickles
CLEANED_PKL = 'pickles/cleaned.pkl'
FEATURED_PKL = 'pickles/featured.pkl'
GROUPME_PKL = 'pickles/groupme.pkl'

TO_PERCENT = {1: 1, 2: 0.1, 3: 0.01}


def ask_question(s):
    return str(input(s)).upper()[0] == 'Y'


def get_input():
    try:
        inn = str(input(INTRO_INPUT))
    except KeyboardInterrupt:
        goodbye()

    return inn.lower().strip()


def goodbye():
    print('\nGoodbye')
    sys.exit()


def main():
    print('\n')
    print(INTRO)
    action = get_input()

    if action == 'groupme':
        print(GROUPME)
        action = get_input()
        if action != '':
            goodbye()

        train, _, test = split(unpickle(GROUPME_PKL))
        ml(train, test, 2)
        goodbye()

    elif action == 'freecodecamp':
        print(FREECODECAMP)
        action = get_input()
        try:
            action = int(action)
        except ValueError:
            goodbye()

        if action not in TO_PERCENT:
            goodbye()

        start = time()

        data = clean(DATA_FILE, TO_PERCENT[action])
        data = data.sample(frac=1.0).reset_index(drop=True)

        logger.info(f'saving to pickle: {CLEANED_PKL}')
        data.to_pickle(CLEANED_PKL)

        data = get_features(data)
        logger.info(f'saving to pickle: {FEATURED_PKL}')
        data.to_pickle(FEATURED_PKL)

        train, _, test = split(data)
        ml(train, test, 10)

        logger.info(f'Time Elapsed: {round(time() - start, 3)}s')

        print('Press enter to quit')
        get_input()
        goodbye()

    else:
        goodbye()

    """
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
    """


if __name__ == "__main__":
    main()
