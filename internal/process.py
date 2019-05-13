# This file preprocess the whole dataset and store them in a json file
import pandas as pd


def process(in_name, out_name):
    """
    TODO: better comment
    this part of the code takes in the original csv, filter and sort a
    list of users who send more than 10000 messages and print out a
    dataframe with messages from users in that list, and dataframe only
    contain columns
    """
    df = pd.read_csv(in_name, na_values=None)
    df = df.dropna(subset=['text'])
    df_count = df[['fromUser.id', 'text']]
    message_count = df_count.groupby(['fromUser.id']).count()
    user_list = list(message_count[message_count['text'] > 10000].index)
    df = df[df['fromUser.id'].isin(user_list)][
        [
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
    ]
    # serializes the dataframe in json file
    with open(out_name, 'w') as output:
        df.to_json(orient='records', path_or_buf=output)
    print('Success!')


if __name__ == '__main__':
    in_file = 'freecodecamp_casual_chatroom.csv'
    out_file = 'data.json'
    process(in_file, out_file)
