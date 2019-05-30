from sklearn.feature_extraction.text import CountVectorizer
from features import get_features
# from gensim import matutils, models
import pandas as pd
import scipy.sparse
import logging
import pickle

import re

DATA_FILE = 'data/freecodecamp_casual_chatroom.csv'
DEBUG = True

logging.basicConfig(level=logging.DEBUG)


def clean(file):
    # drop empty text
    logging.info('reading file into dataframe')
    if not DEBUG:
        df = pd.read_csv(file, na_values=None, low_memory=False)
    else:
        df = pd.read_csv(file, na_values=None, low_memory=False, nrows=5000)
    df = df.dropna(subset=['text'])

    # select certain columns
    df = df[[
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


def clean_sentence(sentence):
    return re.sub(r'[^A-Za-z\s]+', '', sentence.lower())


# broken
def topic_modeling(column):
    sparse_counts = scipy.sparse.csr_matrix(tdm_df)
    corpus = matutils.Sparse2Corpus(sparse_counts)
    cv = pickle.load(open(cv_pickle, "rb"))
    id2word = dict((v, k) for k, v in cv.vocabulary_.items())
    lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=10,
                          passes=100)
    return(lda.print_topics())


# broken
def convert_dtm(clean_data):
    clean_df = pd.read_pickle(clean_data)
    cv = CountVectorizer(stop_words='english')
    data_cv = cv.fit_transform(clean_df.text)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = clean_df['fromUser.id']
    print(data_dtm)
    pickle.dump(cv, open("cv.pkl", "wb"))
    data_dtm.to_pickle('dtm_data.pkl')


# broken
def remove_common_words(pickle):
    df = pd.read_pickle(pickle)
    message_count = df.groupby(['fromUser.id']).count()
    user_list = list(message_count[message_count['text'] > 100000].index)
    df['text'] += " "
    df = df[df['fromUser.id'].isin(user_list)]
    user_df = pd.DataFrame(df.groupby('fromUser.username')['text'].sum())
    cv = CountVectorizer(stop_words='english')
    data_cv = cv.fit_transform(user_df.text)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = user_df.index
    print(data_dtm)
    """
    top_dict = {}
    for c in data_dtm.columns:
        top = data_dtm[c].sort_values(ascending=False).head(30)
        top_dict[c] = list(zip(top.index, top.values))
    print(top_dict)
    """


def main():
    data = clean(DATA_FILE)
    data = get_features(data)
    print(data)
    breakpoint()

    # run main.py to use the pickles
    # print(data)
    # data.to_pickle('data.pkl')  # pickle for future usage


if __name__ == '__main__':
    main()
