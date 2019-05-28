import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob


def clean_sentence(sentence):
    result = sentence.lower()
    result = re.sub('\[!@#$%^&().*?\:"<>~+=]', '', result)
    result = re.sub('[%s]' % re.escape(string.punctuation), '', result)
    result = re.sub('\w*\d\w*', '', result)
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
    df.to_pickle('corpus.pkl')
    df['text'] = df['text'].apply(clean_sentence)
    df.to_pickle('clean_data.pkl')


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


def get_features(pickle):
    df = pd.read_pickle(pickle)
    df['polarity'] = df['text'].apply(pol)
    df['subjectivity'] = df['text'].apply(sub)

    df.to_pickle('featured_data.pkl')


def main():
    clean1('Data/freecodecamp_casual_chatroom.csv')
    # remove_common_words('clean_data.pkl')
    get_features('clean_data.pkl')


if __name__ == '__main__':
    main()
