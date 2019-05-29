from textblob import TextBlob
import pandas as pd


def polarity(df):
    df['polarity'] = df['text_clean'].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    return df['polarity']


def subjectivity(df):
    df['subjectivity'] = df['text_clean'].apply(
        lambda x: TextBlob(x).sentiment.subjectivity
    )
    return df['subjectivity']


def word_count(df):
    df['word_count'] = df['text_clean'].str.split().apply(len)
    return df['word_count']


def avg_word_length(df):
    df['ave_word_length'] = \
        df['text_clean'].str.replace(' ', '').apply(len) / df['word_count']
    return df['ave_word_length']


def get_features(df):
    # sentiment features
    pol_series = polarity(df)
    sub_series = subjectivity(df)
    # word count
    count_series = word_count(df)
    # average word length
    length_series = avg_word_length(df)

    data_dict = {'id': df['id'], pol_series.name: pol_series, sub_series.name: sub_series,
                 count_series.name: count_series, length_series.name:
                 length_series}
    df = pd.DataFrame(data_dict)
    df.to_pickle('feature_data.pkl')


if __name__ == "__main__":
    print(pd.read_pickle('feature_data.pkl'))
