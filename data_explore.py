import pandas as pd
import numpy as np

feature_columns = ['polarity', 'subjectivity',
                   'wordCount', 'avgWordLength',
                   'adjRatio', 'verbRatio', 'nounRatio',
                   'mentionsCount', 'urlsCount']
data = pd.read_pickle('data.pkl')
clean = pd.read_pickle('cleaned.pkl')
feature = pd.read_pickle('featured.pkl')


def convert(x):
    r = x[11:16]
    h = int(r[0:2])
    m = int(r[3:5])
    t = h + m / 100
    return t


feature['sent'] = feature['sent'].apply(convert)
print(feature)
feature.to_pickle('time_data.pkl')

# feature = feature[feature.columns != 'sent']
print(feature)
#feature.to_pickle('featured.pkl')
