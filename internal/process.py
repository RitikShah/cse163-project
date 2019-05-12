# This file preprocess the whole dataset and store them in a json file
import pandas as pd
import json


# Some constants for the file names
def process(in_name, out_name):
    """
    TODO: better comment
    this part of the code takes in the original csv, filter and sort a
    list of users who send more than 10000 messages and print out a
    dataframe with messages from users in that list, and dataframe only
    contain columns
    """
    df = pd.read_csv(in_name)
    df_filtered = df[['fromUser.displayName', 'fromUser.id', 'text']]
    rank = df_filtered.groupby(['fromUser.displayName', 'fromUser.id']).count()
    rank = rank[rank['text'] > 10000]
    sort_rank = rank.sort_values(ascending=False, by='text')
    # print(sort_rank)  # TODO: use logger
    name_list = sort_rank.index.tolist()
    # print(name_list, len(name_list)) # TODO: use logger
    df = df[df['fromUser.displayName'].isin(name_list)]
    col_list = [
        'fromUser.displayName',
        'fromUser.username',
        'fromUser.id',
        'mentions',
        'urls',
        'readBy',
        'editedAt',
        'sent',
        'id',
        'text'
    ]
    df = df[col_list]
    # print(df)  # TODO: use logger

    # serializes the dataframe in json file
    with open(in_name, 'w') as file:
        json.dump(df.to_json(orient='records'), file, indent=1)
    print('Success!')  # TODO: use Logger


if __name__ == '__main__':
    in_file = 'data/freecodecamp_casual_chatroom.csv'
    out_file = 'data/data.json'
    process(in_file, out_file)
