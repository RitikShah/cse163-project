from textblob import TextBlob
import numpy as np
import logging

logger = logging.getLogger(__name__)

ADJS = r'(JJR)|(JJS)|(JJ)'
VERBS = r'(VBD)|(VBG)|(VBN)|(VBP)|(VBZ)|(VB)'
NOUNS = r'(NNPS)|(NNP)|(NNS)|(NN)'


def transform(df, col):
    return df[col].apply(TextBlob)


def polarity(col):
    return col.apply(lambda x: x.sentiment.polarity)


def subjectivity(col):
    return col.apply(lambda x: x.sentiment.subjectivity)


def word_count(col):
    return col.str.count(r'\s')


def avg_word_length(df):
    return df['text'].str.replace(' ', '').str.len() / df['wordCount']


# could use lambdas, but using internal functions bc readability
def adj_ratio(df):
    return df['textblobs'].str.count(ADJS) / df['wordCount']


def verb_ratio(df):
    return df['textblobs'].str.count(VERBS) / df['wordCount']


def noun_ratio(df):
    return df['textblobs'].str.count(NOUNS) / df['wordCount']


def mentions(col):
    return col.apply(eval).apply(len)


def urls(col):
    return col.apply(eval).apply(len)


def exclamations(col):
    return col.str.count('!')


def questions(col):
    return col.str.count(r'\?')  # it's trying to parse it as regex :/


def fix_infs(df, col):
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(value=0)


def get_features(features):
    logger.info('getting features from text and text_clean')
    features = features[['text', 'text_clean', 'mentions',
                         'urls', 'id', 'readBy']]  # expand on
    feature_columns = ['polarity', 'subjectivity',
                       'wordCount', 'avgWordLength',
                       'adjRatio', 'verbRatio', 'nounRatio',
                       'mentionsCount', 'urlsCount',
                       'exclamationCount', 'questionCount']

    # making textblobs
    logger.debug('preprocessing textblobs')
    logger.debug('  textblobs for normal text')
    features['textblobs'] = transform(features, 'text').apply(
        lambda blob: ' '.join([x[1] for x in blob.tags])
    )
    logger.debug('  textblobs for clean text')
    features['textblobs_clean'] = transform(features, 'text_clean')

    logger.debug('calculating polarity and subjectivity')
    # sentiment features
    logger.debug('  pol')
    features['polarity'] = polarity(features['textblobs_clean'])
    logger.debug('  sub')
    features['subjectivity'] = subjectivity(features['textblobs_clean'])

    logger.debug('calculating word count')
    # word count
    features['wordCount'] = word_count(features['text'])

    # average word length
    logger.debug('calculating avg word length')
    features['avgWordLength'] = avg_word_length(features)
    fix_infs(features, 'avgWordLength')

    logger.debug('calculating ratios')
    # ratios
    logger.debug('  adj')
    features['adjRatio'] = adj_ratio(features)
    fix_infs(features, 'adjRatio')

    logger.debug('  verb')
    features['verbRatio'] = verb_ratio(features)
    fix_infs(features, 'verbRatio')

    logger.debug('  noun')
    features['nounRatio'] = noun_ratio(features)
    fix_infs(features, 'nounRatio')

    logger.debug('mentions and url count')
    features['mentionsCount'] = mentions(features['mentions'])
    features['urlsCount'] = urls(features['urls'])

    logger.debug('exclamation and question counts')
    features['exclamationCount'] = exclamations(features['text'])
    features['questionCount'] = questions(features['text'])

    logger.debug('done!')
    return features.loc[:, feature_columns + ['id', 'readBy']]


if __name__ == '__main__':
    ''' testing the code '''
    import clean_data
    data = clean_data.clean(clean_data.DATA_FILE)
    out = get_features(data)
    logger.info('  done!')
    # print(out)
    # breakpoint()
