import pandas as pd

df = pd.read_csv('freecodecamp_casual_chatroom.csv')
df_filtered = df[['fromUser.displayName', 'text']]
rank = df_filtered.groupby(['fromUser.displayName']).count()
rank = rank[rank['text'] > 10000]
rank = rank.sort_values(ascending=False, by='text')
name_list = rank.index.tolist()
df = df[df['fromUser.displayName'].isin(name_list)]
col_list = ['fromUser.displayName', 'id', 'mentions', 'readBy', 'sent', 'text']
df = df[col_list]
print(df)
"""
this part of the code takes in the original csv, filter and sort a list of users
who send more than 10000 messages and print out a dataframe with messages from 
users in that list, and dataframe only contain columns 
