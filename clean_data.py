import pandas as pd
import re
import string

# drop empty text
df = pd.read_csv('Data/freecodecamp_casual_chatroom.csv', na_values=None,
                 low_memory=False)
df = df.loc[0:10]
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


def clean_data(sentence):
    result = sentence.lower()
    result = re.sub('\[!@#$%^&().*?\:"<>~+=]', '', result)
    result = re.sub('[%s]' % re.escape(string.punctuation), '', result)
    result = re.sub('\w*\d\w*', '', result)
    return result


df['text'] = df['text'].apply(clean_data)
print(df)
