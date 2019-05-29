import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from gensim import matutils, models
import scipy.sparse
import pickle


def clean_sentence(sentence):
    result = sentence.lower()
    result = re.sub(r'\[!@#$%^&().*?\:"<>~+=]', '', result)
    result = re.sub(r'[%s]' % re.escape(string.punctuation), '', result)
    result = re.sub(r'\w*\d\w*', '', result)
    return result


def clean1(file):
    # drop empty text
    df = pd.read_csv(file, na_values=None, low_memory=False)
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
    df = df.loc[0:1000, :]
    df.to_pickle('corpus.pkl')
    df['text'] = df['text'].apply(clean_sentence)
    df.to_pickle('clean_data.pkl')


def convert_dtm(clean_data):
    clean_df = pd.read_pickle(clean_data)
    cv = CountVectorizer(stop_words='english')
    data_cv = cv.fit_transform(clean_df.text)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = clean_df['fromUser.id']
    print(data_dtm)
    pickle.dump(cv, open("cv.pkl", "wb"))
    data_dtm.to_pickle('dtm_data.pkl')


def get_features(clean_pickle, og_pickel, dtm_pickel, cv_pickle):
    df = pd.read_pickle(og_pickel)
    features_df = pd.read_pickle(clean_pickle)
    dtm_df = pd.read_pickle(dtm_pickel)
    # sentiment features
    features_df['polarity'] = df['text'].apply(pol)
    features_df['subjectivity'] = df['text'].apply(sub)
    # word count
    features_df['wordCount'] = df['text'].str.split().apply(len)
    # average word length
    features_df['ave_word_length'] = df['text'].str.replace(" ", ''). \
        apply(len) / features_df['wordCount']
    features_df.to_pickle('featured_data.pkl')
    # topic
    tdm_df = dtm_df.transpose()
    sparse_counts = scipy.sparse.csr_matrix(tdm_df)
    corpus = matutils.Sparse2Corpus(sparse_counts)
    cv = pickle.load(open(cv_pickle, "rb"))
    id2word = dict((v, k) for k, v in cv.vocabulary_.items())
    lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=10,
                          passes=100)
    print(lda.print_topics())
    # Ajective ratio


def topic_modeling(column):
    sparse_counts = scipy.sparse.csr_matrix(tdm_df)
    corpus = matutils.Sparse2Corpus(sparse_counts)
    cv = pickle.load(open(cv_pickle, "rb"))
    id2word = dict((v, k) for k, v in cv.vocabulary_.items())
    lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=10,
                          passes=100)
    return(lda.print_topics())


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


def pol(x):
    return TextBlob(x).sentiment.polarity


def sub(x):
    return TextBlob(x).sentiment.subjectivity


def main():
    # clean1('Data/freecodecamp_casual_chatroom.csv')
    # remove_common_words('clean_data.pkl')
    # convert_dtm('clean_data.pkl')
    get_features('clean_data.pkl', 'corpus.pkl', 'dtm_data.pkl', 'cv.pkl')


if __name__ == '__main__':
    main()
