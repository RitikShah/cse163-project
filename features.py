from textblob import TextBlob


def polarity(df):
    df['polarity'] = df['text_clean'].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )


def subjectivity(df):
    df['subjectivity'] = df['text_clean'].apply(
        lambda x: TextBlob(x).sentiment.subjectivity
    )


def word_count(df):
    df['word_count'] = df['text_clean'].str.split().apply(len)


def avg_word_length(df):
    df['ave_word_length'] = \
        df['text_clean'].str.replace(' ', '').apply(len) / df['word_count']


def get_features(df):
    # sentiment features
    polarity(df)
    subjectivity(df)

    # word count
    word_count(df)

    # average word length
    avg_word_length(df)
