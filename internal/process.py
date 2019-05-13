# This file preprocess the whole dataset and store them in a json file
import pandas as pd
import json


def process(in_name, out_name):
    """
    TODO: better comment
    this part of the code takes in the original csv, filter and sort a
    list of users who send more than 10000 messages and print out a
    dataframe with messages from users in that list, and dataframe only
    contain columns
    """
    try:
        df = pd.read_csv(in_name)
        df['text_count'] = \
            df.groupby(['fromUser.displayName', 'fromUser.id']).count()
        df = df[df['text_count'] > 10000][
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
        # print(df)  # TODO: use logger

        # serializes the dataframe in json file
        df.to_json(orient='records', path_or_buf=out_name)
        print('Success!')  # TODO: use Logger
    except Exception as e:
        print(e)
        breakpoint()


if __name__ == '__main__':
    in_file = 'data/freecodecamp_casual_chatroom.csv'
    out_file = 'data/data.json'
    process(in_file, out_file)
