from random import random
import pandas as pd
import logging
import re

DATA_FILE = 'data/freecodecamp_casual_chatroom_anon.csv'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This optimizes the data frames as it puts in specific types for each of the
#  column. The dataframe automatically assumes that everything is an object.
# This also used to only read in certain columns which I believe improves the
#  it takes to read in the table.
COL_TYPES = {
    'fromUser.id': str,
    'mentions.userIds': object,
    'urls': object,
    'readBy': int,
    'id': str,
    'text': str
}


def clean_sentence(sentence):
    """ Regex to remove junk from a sentence (used in sentiment analysis) """
    return re.sub(r'[^A-Za-z\s]+', '', sentence.lower())


def clean(file, percent):
    """ Prepares data from read in csv file """
    logger.info('reading file into dataframe')
    if percent == 1.0:
        df = pd.read_csv(file, na_values=None, dtype=COL_TYPES,
                         low_memory=False, usecols=COL_TYPES.keys())
    else:
        # this randomly grabs a percentage of the data
        df = pd.read_csv(file, na_values=None, low_memory=False,
                         dtype=COL_TYPES, usecols=COL_TYPES.keys(),
                         skiprows=lambda i: i > 0 and random() > percent)
    df['text'] = df['text'].fillna(value='')

    logger.info('cleaning text into text_clean')
    df['mentions'] = df['mentions.userIds'].apply(
        lambda x: len(eval(x))
    )

    df['urls'] = df['urls'].apply(
        lambda x: len(eval(x))
    )

    df['text_clean'] = df['text'].apply(clean_sentence)
    return df


def main():
    """ Runs functions to test clean_data and saves it in a pkl """
    data = clean(DATA_FILE)
    logger.info('pickling to cleaned.pkl')
    data.to_pickle('cleaned.pkl')


if __name__ == '__main__':
    main()
