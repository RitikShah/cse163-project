import pandas as pd
import numpy as np

feature_columns = ['polarity', 'subjectivity',
                   'wordCount', 'avgWordLength',
                   'adjRatio', 'verbRatio', 'nounRatio',
                   'mentionsCount', 'urlsCount']

data = pd.read_pickle('data.pkl')
print(data)
data[data.isin([np.nan, np.inf, -np.inf]).any()][feature_columns] = 0
print(data)
data.to_pickle('data.pkl')
