from textblob import TextBlob
import nltk


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


def adj_ratio(sentence):
    word_list = sentence.split()
    if len(word_list) == 0:
        return 0
    tagged_list = nltk.pos_tag(word_list)
    adj_count = 0
    for word in tagged_list:
        if (word[1] == 'JJ') | (word[1] == 'JJR') | (word[1] == 'JJS'):
            adj_count += 1
    return adj_count / len(tagged_list)


def verb_ratio(sentence):
    word_list = sentence.split()
    if len(word_list) == 0:
        return 0
    tagged_list = nltk.pos_tag(word_list)
    verb_count = 0
    for word in tagged_list:
        if (word[1] == 'VB') | (word[1] == 'VBD') | (word[1] == 'VBG') | \
           (word[1] == 'VBN') | (word[1] == 'VBP') | (word[1] == 'VBZ'):
            verb_count += 1
    return verb_count / len(tagged_list)


def noun_ratio(sentence):
    data.to_pickle('data.pkl')  # pickle for future usage
    if len(word_list) == 0:
        return 0
    tagged_list = nltk.pos_tag(word_list)
    noun_count = 0
    for word in tagged_list:
        if (word[1] == 'NN') | (word[1] == 'NNS') | (word[1] == 'NNP') | \
           (word[1] == 'NNPS'):
            noun_count += 1
    return noun_count / len(tagged_list)


def get_features(df):
    # sentiment features
    polarity(df)
    subjectivity(df)

    # word count
    word_count(df)

    # average word length
    avg_word_length(df)
