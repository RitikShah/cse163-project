from textblob import TextBlob
import numpy as np
import logging

ADJS = {'JJ', 'JJR', 'JJS'}
VERBS = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
NOUNS = {'NN', 'NNS', 'NNP', 'NNPS'}  # unused


def polarity(df):
    return df['text_clean'].apply(lambda x: TextBlob(x).sentiment.polarity)


def subjectivity(df):
    return df['text_clean'].apply(lambda x: TextBlob(x).sentiment.subjectivity)


def word_count(df):
    return df['text'].str.split().apply(len)


def avg_word_length(df):
    return df['text'].str.replace(' ', '').apply(len) / df['wordCount']


# could use lambdas, but using internal functions bc readability
def adj_ratio(df):
    def _calc(sen):
        return len(list(filter(lambda x: x[1] in ADJS, TextBlob(sen).tags)))

    return df['text'].apply(_calc) / df['wordCount']


def verb_ratio(df):
    def _calc(sen):
        return len(list(filter(lambda x: x[1] in VERBS, TextBlob(sen).tags)))

    return df['text'].apply(_calc) / df['wordCount']


def noun_ratio(df):
    def _calc(sen):
        return len(TextBlob(sen).noun_phrases)

    return df['text'].apply(_calc) / df['wordCount']


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

    # sentiment features
    features['polarity'] = polarity(features)
    features['subjectivity'] = subjectivity(features)

    # word count
    features['wordCount'] = word_count(features)

    # average word length
    features['avgWordLength'] = avg_word_length(features)
    fix_infs(features, 'avgWordLength')

    # ratios
    features['adjRatio'] = adj_ratio(features)
    fix_infs(features, 'adjRatio')
    features['verbRatio'] = verb_ratio(features)
    fix_infs(features, 'verbRatio')
    features['nounRatio'] = noun_ratio(features)
    fix_infs(features, 'nounRatio')

    features['mentionsCount'] = mentions(features)
    features['urlsCount'] = urls(features)

    return features.loc[:, feature_columns + ['id', 'readBy']]


if __name__ == '__main__':
    ''' testing the code '''
    import clean_data
    data = clean_data.clean(clean_data.DATA_FILE)
    out = get_features(data)
    logging.info('  done!')
    # print(out)
    # breakpoint()
