from textblob import TextBlob
import numpy as np
import logging

ADJS = {'JJ', 'JJR', 'JJS'}
VERBS = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
NOUNS = {'NN', 'NNS', 'NNP', 'NNPS'}  # unused


def transform(df, col):
    return df[col].apply(TextBlob)


def polarity(df):
    return df['textblobs_clean'].apply(lambda x: x.sentiment.polarity)


def subjectivity(df):
    return df['textblobs_clean'].apply(lambda x: x.sentiment.subjectivity)


def word_count(df):
    return df['text'].str.split().apply(len)


def avg_word_length(df):
    return df['text'].str.replace(' ', '').apply(len) / df['wordCount']


# could use lambdas, but using internal functions bc readability
def adj_ratio(df):
    def _calc(sen):
        return sum(
            1 for v in filter(lambda x: x[1] in ADJS, sen.tags)
        )

    return df['textblobs'].apply(_calc) / df['wordCount']


def verb_ratio(df):
    def _calc(sen):
        return sum(
            1 for v in filter(lambda x: x[1] in VERBS, sen.tags)
        )

    return df['textblobs'].apply(_calc) / df['wordCount']


def noun_ratio(df):
    def _calc(sen):
        return len(sen.noun_phrases)

    return df['textblobs'].apply(_calc) / df['wordCount']


def mentions(df):
    return df['mentions'].apply(eval).apply(len)


def urls(df):
    return df['urls'].apply(eval).apply(len)


def fix_infs(df, col):
    df.loc[np.isinf(df[col]), col] = 0


def get_features(df):
    logging.info('getting features from text and text_clean')
    features = df.copy()  # deep copy
    feature_columns = ['polarity', 'subjectivity',
                       'wordCount', 'avgWordLength',
                       'adjRatio', 'verbRatio', 'nounRatio',
                       'mentionsCount', 'urlsCount']

    # making textblobs
    logging.debug('preprocessing textblobs')
    features['textblobs'] = transform(df, 'text')
    features['textblobs_clean'] = transform(df, 'text_clean')

    logging.debug('calculating polarity and subjectivity')
    # sentiment features
    logging.debug('  pol')
    features['polarity'] = polarity(features)
    logging.debug('  sub')
    features['subjectivity'] = subjectivity(features)

    logging.debug('calculating word count')
    # word count
    features['wordCount'] = word_count(features)

    # average word length
    logging.debug('calculating avg word length')
    features['avgWordLength'] = avg_word_length(features)
    fix_infs(features, 'avgWordLength')

    logging.debug('calculating ratios')
    # ratios
    logging.debug('  adj')
    features['adjRatio'] = adj_ratio(features)
    fix_infs(features, 'adjRatio')

    logging.debug('  verb')
    features['verbRatio'] = verb_ratio(features)
    fix_infs(features, 'verbRatio')

    logging.debug('  noun')
    features['nounRatio'] = noun_ratio(features)
    fix_infs(features, 'nounRatio')

    logging.debug('mentions and url count')
    features['mentionsCount'] = mentions(features)
    features['urlsCount'] = urls(features)

    logging.debug('done!')
    return features.loc[:, feature_columns + ['id', 'readBy']]


if __name__ == '__main__':
    ''' testing the code '''
    import clean_data
    data = clean_data.clean(clean_data.DATA_FILE)
    out = get_features(data)
    logging.info('  done!')
    # print(out)
    # breakpoint()
