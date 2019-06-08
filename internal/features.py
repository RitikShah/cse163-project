from .utils import unpickle

from textblob import TextBlob
import pandas as pd
import numpy as np
import logging

ADJS = r'(JJR)|(JJS)|(JJ)'
VERBS = r'(VBD)|(VBG)|(VBN)|(VBP)|(VBZ)|(VB)'
NOUNS = r'(NNPS)|(NNP)|(NNS)|(NN)'


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None  # default='warn' annoying :/


def transform(df, col):
    return df[col].apply(TextBlob)


def polarity(df):
    return df['textblobs_clean'].apply(lambda x: x.sentiment.polarity)


def subjectivity(df):
    return df['textblobs_clean'].apply(lambda x: x.sentiment.subjectivity)


def word_count(df):
    return df['text'].str.count(r'\s')


def avg_word_length(df):
    return df['text'].str.replace(' ', '').str.len() / df['wordCount']


# could use lambdas, but using internal functions bc readability
def adj_ratio(df):
    return df['textblobs'].str.count(ADJS) / df['wordCount']


def verb_ratio(df):
    return df['textblobs'].str.count(VERBS) / df['wordCount']


def noun_ratio(df):
    return df['textblobs'].str.count(NOUNS) / df['wordCount']


def mentions(df):
    return df['mentions'].apply(eval).apply(len)


def urls(df):
    return df['urls'].apply(eval).apply(len)


def exclamations(df):
    return df['text'].str.count('!')


def questions(df):
    return df['text'].str.count(r'\?')  # it's trying to parse it as regex :/


def fix_infs(df, col):
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(value=0)


def get_features(data):
    logger.info('getting features from text and text_clean')
    features = data[['text', 'text_clean', 'mentions',
                     'urls', 'id', 'readBy', 'fromUser.id']]  # expand on
    del data
    feature_columns = ['polarity', 'subjectivity',
                       'wordCount', 'avgWordLength',
                       'adjRatio', 'verbRatio', 'nounRatio',
                       'mentionsCount', 'urlsCount',
                       'exclamationCount', 'questionCount']

    # making textblobs
    logger.info('preprocessing textblobs')
    logger.info('  textblobs for normal text')
    features['textblobs'] = transform(features, 'text').apply(
        lambda blob: ' '.join([x[1] for x in blob.tags])  # getting only tags
    )
    logger.info('  textblobs for clean text')
    features['textblobs_clean'] = transform(features, 'text_clean')

    logger.info('calculating polarity and subjectivity')
    # sentiment features
    logger.info('  pol')
    features['polarity'] = polarity(features)
    logger.info('  sub')
    features['subjectivity'] = subjectivity(features)

    logger.info('calculating word count')
    # word count
    features['wordCount'] = word_count(features)

    # average word length
    logger.info('calculating avg word length')
    features['avgWordLength'] = avg_word_length(features)
    fix_infs(features, 'avgWordLength')

    logger.info('calculating ratios')
    # ratios
    logger.info('  adj')
    features['adjRatio'] = adj_ratio(features)
    fix_infs(features, 'adjRatio')

    logger.info('  verb')
    features['verbRatio'] = verb_ratio(features)
    fix_infs(features, 'verbRatio')

    logger.info('  noun')
    features['nounRatio'] = noun_ratio(features)
    fix_infs(features, 'nounRatio')

    logger.info('mentions and url count')
    features['mentionsCount'] = mentions(features)
    features['urlsCount'] = urls(features)

    logger.info('exclamation and question counts')
    features['exclamationCount'] = exclamations(features)
    features['questionCount'] = questions(features)

    logger.info('done!')
    return features.loc[:, feature_columns + ['id', 'readBy', 'fromUser.id']]


if __name__ == '__main__':
    ''' testing the code '''
    data = unpickle('pickles/cleaned.pkl')
    out = get_features(data)
    logger.info('  done!')
    # data.to_pickle('pickles/bleh.pkl')
    # print(out)
