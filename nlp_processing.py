import pandas as pd
import re
import string
import json

df = None
with open('data.json') as j:
    data = json.load(j)
    df = pd.read_json(data, orient='records')

print(len(df))

'''
for i in range(len(df)):
    if df.loc[i, 'text'] == None:
        print(df.loc[i, 'text'])
'''


def clean_data(text):
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


df.loc[:, 'text'] = df['text'].apply(clean_data)
print(df)

# print(type(df.text[0]))


df['text'].apply(clean_data)
print(df)
# df['sent'] = df['sent'].lower()
# df['sent'] = df['sent'].sub('[%s]' % re.escape(string.punctuation), '', text)
# print(df)
