import pandas as pd
import json


"""
this part of the code takes in the original csv, filter and sort a
list of users who send more than 10000 messages and print out a
dataframe with messages from users in that list, and dataframe only
contain columns:'fromUser.displayName', 'fromUser.id', 'mentions',
'readBy', 'sent', 'text'
"""
df = pd.read_csv('freecodecamp_casual_chatroom.csv', na_values=None)
df = df.dropna(subset=['text'])
df_filtered = df[['fromUser.displayName', 'text']]
rank = df_filtered.groupby(['fromUser.displayName']).count()
rank = rank[rank['text'] > 10000]
sort_rank = rank.sort_values(ascending=False, by='text')
print(sort_rank)
name_list = sort_rank.index.tolist()
print(name_list, len(name_list))
df = df[df['fromUser.displayName'].isin(name_list)]
col_list = ['fromUser.displayName', 'fromUser.id', 'mentions',
            'readBy', 'sent', 'text']
df = df[col_list]
print(df)

"""
this part convert filtered dataframe into json object and save json
object into json file
"""
js = df.to_json(orient='records')
json.dumps(js, js_file)
