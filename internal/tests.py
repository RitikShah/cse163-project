from .features import get_features
from .clean_data import clean
from .utils import unpickle

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_FILE = 'data/freecodecamp_casual_chatroom_anon.csv'

FEATURED_PKL = 'pickles/featured.pkl'
CLEANED_PKL = 'pickles/cleaned.pkl'


def assert_equals(expected, recieved):
    """ Compares expected and recieved: throws a message if not equal """
    try:
        assert(expected == recieved)
        return True
    except AssertionError:
        print(f'**   FAILED: expected: {expected}, recieved: {recieved}')
        return False


def test_cleaned_data():
    """ Testing the cleaning of data """
    print('* Testing the cleaning of data')
    passed = 0

    print('*   testing reading data')
    data = clean(DATA_FILE, .01)
    passed += 1

    print('*   testing columns')
    cols = {'fromUser.id', 'mentions', 'urls',
            'readBy', 'id', 'text', 'text_clean'}

    if assert_equals(set(data.columns), cols):
        passed += 1

    if passed == 2:
        print('* Passed all cases ')
    else:
        print(f'* Failed {2 - passed} cases')

    return data


def test_features(data):
    """ Testing the features of data """
    print('* Testing the text processing of features')
    passed = 0

    print('*   testing getting features')
    data = get_features(data)
    passed += 1

    print('*   testing columns')
    cols = {
        'mentionsCount', 'urlsCount', 'exclamationCount', 'questionCount',
        'nounRatio', 'verbRatio', 'adjRatio', 'avgWordLength', 'wordCount',
        'polarity', 'subjectivity', 'id', 'fromUser.id', 'readBy'
    }

    if assert_equals(set(data.columns), cols):
        passed += 1

    print('*   testing subjecivity and polarity ranges')
    sub_max = assert_equals(len(data[data['subjectivity'] > 1]), 0)
    pol_max = assert_equals(len(data[data['polarity'] > 1]), 0)
    sub_min = assert_equals(len(data[data['subjectivity'] < -1]), 0)
    pol_min = assert_equals(len(data[data['polarity'] < -1]), 0)

    if sub_max and sub_min and pol_max and pol_min:
        passed += 1

    print('*   testing ratio ranges')
    verb_max = assert_equals(len(data[data['verbRatio'] > 1]), 0)
    verb_min = assert_equals(len(data[data['verbRatio'] < 0]), 0)
    adj_max = assert_equals(len(data[data['adjRatio'] > 1]), 0)
    adj_min = assert_equals(len(data[data['adjRatio'] < 0]), 0)
    noun_max = assert_equals(len(data[data['nounRatio'] > 1]), 0)
    noun_min = assert_equals(len(data[data['nounRatio'] < 0]), 0)

    if verb_max and verb_min and adj_max and adj_min and noun_max and noun_min:
        passed += 1

    print('*  testing other non-negative features')
    words = assert_equals(len(data[data['wordCount'] < 0]), 0)
    length = assert_equals(len(data[data['avgWordLength'] < 0]), 0)
    mentions = assert_equals(len(data[data['mentionsCount'] < 0]), 0)
    urls = assert_equals(len(data[data['urlsCount'] < 0]), 0)
    exclaims = assert_equals(len(data[data['exclamationCount'] < 0]), 0)
    questions = assert_equals(len(data[data['questionCount'] < 0]), 0)

    if words and length and mentions and urls and exclaims and questions:
        passed += 1

    if passed == 5:
        print('* Passed all cases ')
    else:
        print(f'* Failed {5 - passed} cases')

    return data


def playground():
    """ Throws you into a interpreter loop with data loaded in """
    import pandas as pd  # noqa
    data = unpickle(FEATURED_PKL)  # noqa
    logger.info('entering breakpoint')
    logger.info('access data in var: `data`')
    logger.info('and pandas as `pd`')
    breakpoint()


def main():
    """ Running the entire suite of tests """
    print('* *** Running all tests *** *')
    print()
    data = test_cleaned_data()
    data = test_features(data)
    print()
    print('* *** All tests passed *** *')


if __name__ == '__main__':
    main()
