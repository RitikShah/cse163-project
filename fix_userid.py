import pandas as pd


clean = pd.read_pickle('pickle/5m_pkls/cleaned.pkl')
feature = pd.read_pickle('pickle/5m_pkls/featured.pkl')

full_data = clean.merge(feature, left_on='id', right_on='id', how='outer')
full_data.to_pickle('pickle/5m_pkls/full_data.pkl')

feature = full_data.loc[:, ['polarity', 'subjectivity',
                            'wordCount', 'avgWordLength',
                            'adjRatio', 'verbRatio', 'nounRatio',
                            'mentionsCount', 'urlsCount',
                            'exclamationCount', 'questionCount'
                            'fromUser.id']]
feature.to_pickle('pickle/5m_pkls/featured.pkl')
