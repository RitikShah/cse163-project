from .utils import unpickle

from textblob import TextBlob
import pandas as pd
import numpy as np
import logging

# regex to find those specific tags. used in the ration methods
ADJS = r'JJ'
VERBS = r'VB'
NOUNS = r'NN'


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None  # this turns off some warnings


def transform(df, col):
    return df[col].apply(TextBlob)


def polarity(df):
    return df['textblobs'].apply(lambda x: x.sentiment.polarity)


def subjectivity(df):
    return df['textblobs'].apply(lambda x: x.sentiment.subjectivity)


def word_count(df):
    return df['text_clean'].str.count(r'\w')  # counting words


def avg_word_length(df):
    return df['text_clean'].str.replace(' ', '').str.len() / df['wordCount']


def adj_ratio(df):
    return df['tags'].str.count(ADJS) / df['wordCount']


def verb_ratio(df):
    return df['tags'].str.count(VERBS) / df['wordCount']


def noun_ratio(df):
    return df['tags'].str.count(NOUNS) / df['wordCount']


def mentions(df):
    return df['mentions'].apply(eval).apply(len)  # this is dirty but it works


def urls(df):
    return df['urls'].apply(eval).apply(len)


def exclamations(df):
    return df['text'].str.count('!')


def questions(df):
    return df['text'].str.count(r'\?')  # it's trying to parse it as regex :/


def fix_infs(df, col):
    """ Fixes infinite values that could come from division by 0 """
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(value=0)


def get_features(data):
    """ Creates a set of features from a given text """
    logger.info('getting features from text and text_clean')
    features = data[['text', 'text_clean', 'mentions',
                     'urls', 'id', 'readBy', 'fromUser.id']]  # expand on
    del data

    # making textblobs
    logger.info('preprocessing textblobs')
    logger.info('  textblobs for normal text')
    # working with strings instead of objects is much faster for pandas
    # Here, i make a string with tags seperated with spaces
    # This allows me to use regex counting methods which are vectorized (spd!)
    logger.info('  textblobs for text processing')
    features['textblobs'] = transform(features, 'text_clean')
    features['tags'] = features['textblobs'].apply(
        lambda blob: ' '.join([x[1] for x in blob.tags])  # getting only tags
    )

    logger.info('calculating polarity and subjectivity')
    # sentiment features
    logger.info('  polarity')
    features['polarity'] = polarity(features)
    logger.info('  subjectivity')
    features['subjectivity'] = subjectivity(features)

    logger.info('calculating word count')
    features['wordCount'] = word_count(features)

    logger.info('calculating avg word length')
    features['avgWordLength'] = avg_word_length(features)
    fix_infs(features, 'avgWordLength')

    logger.info('calculating ratios')
    logger.info('  adjective')
    features['adjRatio'] = adj_ratio(features)
    fix_infs(features, 'adjRatio')

    logger.info('  verb')
    features['verbRatio'] = verb_ratio(features)
    fix_infs(features, 'verbRatio')

    logger.info('  noun')
    features['nounRatio'] = noun_ratio(features)
    fix_infs(features, 'nounRatio')

    logger.info('mentions and url count')
    features['mentionsCount'] = features['mentions']
    features['urlsCount'] = features['urls']

    logger.info('exclamation and question counts')
    features['exclamationCount'] = exclamations(features)
    features['questionCount'] = questions(features)

    logger.info('done!')

    blacklist = ['text', 'textblobs', 'tags', 'mentions', 'urls', 'text_clean']

    return features.drop(blacklist, axis=1)  # drops columns


if __name__ == '__main__':
    ''' testing the code '''
    data = unpickle('pickles/cleaned.pkl')
    out = get_features(data)
