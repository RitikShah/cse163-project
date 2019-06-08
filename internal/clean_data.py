from random import random
import pandas as pd
import logging
import re

DATA_FILE = 'data/freecodecamp_casual_chatroom.csv'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COL_TYPES = {
    'fromUser.id': str,
    'mentions': object,
    'urls': object,
    'readBy': int,
    'id': str,
    'text': str
}


def clean_sentence(sentence):
    return re.sub(r'[^A-Za-z\s]+', '', sentence.lower())


def clean(file, percent):
    # drop empty text
    logger.info('reading file into dataframe')
    if percent == 1.0:
        df = pd.read_csv(file, na_values=None, dtype=COL_TYPES,
                         low_memory=False)
    else:
        # this randomly grabs a percentage of the data
        df = pd.read_csv(file, na_values=None, low_memory=False,
                         dtype=COL_TYPES,
                         skiprows=lambda i: i > 0 and random() > percent)
    df = df.dropna(subset=['text'])

    # select certain columns
    df = df[list(COL_TYPES.keys())]
    logger.info('cleaning text into text_clean')
    df['text_clean'] = df['text'].apply(clean_sentence)
    return df


def main():
    data = clean(DATA_FILE)
    logger.info('pickling to cleaned.pkl')
    data.to_pickle('cleaned.pkl')


if __name__ == '__main__':
    main()
