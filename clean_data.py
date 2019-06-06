import pandas as pd
import logging
import re

DATA_FILE = 'data/freecodecamp_casual_chatroom.csv'
DEBUG = False

logging.basicConfig(level=logging.DEBUG)


def clean_sentence(sentence):
    return re.sub(r'[^A-Za-z\s]+', '', sentence.lower())


def clean(file):
    # drop empty text
    logging.info('reading file into dataframe')
    if not DEBUG:
        df = pd.read_csv(file, na_values=None, low_memory=False)
    else:
        df = pd.read_csv(file, na_values=None, low_memory=False,
                         nrows=500000)
    df = df.dropna(subset=['text'])

    # select certain columns
    df = df[[
                'sent',
                'fromUser.displayName',
                'fromUser.username',
                'fromUser.id',
                'mentions',
                'urls',
                'readBy',
                'editedAt',
                'id',
                'text'
            ]]
    logging.info('cleaning text into text_clean')
    df['text_clean'] = df['text'].apply(clean_sentence)
    return df


def main():
    data = clean(DATA_FILE)
    logging.info('pickling to cleaned.pkl')
    data.to_pickle('cleaned.pkl')


if __name__ == '__main__':
    main()
