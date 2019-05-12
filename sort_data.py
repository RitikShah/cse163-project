# This file preporcess the whole dataset and store them in a json file
import pandas as pd
import json
import os

# Some constants for the file names
IN_FILE = 'freecodecamp_casual_chatroom.csv'
OUT_FILE = 'data.json'

with open(OUT_FILE, 'w') as js_file:
    if os.stat(OUT_FILE).st_size == 0:
        df = pd.read_csv(IN_FILE)
        df_filtered = df[['fromUser.displayName', 'text']]
        rank = df_filtered.groupby(['fromUser.displayName']).count()
        rank = rank[rank['text'] > 10000]  # Only people w/ more than 1000 msgs
        rank = rank.sort_values(ascending=False, by='text')
        name_list = rank.index.tolist()
        df = df[df['fromUser.displayName'].isin(name_list)]
        col_list = ['fromUser.displayName', 'id', 'mentions', 'readBy',
                    'sent', 'text']
        df = df[col_list]
        print(df)
        """
        this part of the code takes in the original csv, filter and sort a
        list of users who send more than 10000 messages and print out a
        dataframe with messages from users in that list, and dataframe only
        contain columns
        """
        js = df.to_json(orient='records')
        json.dump(js, js_file)
        """
        this part convert filtered dataframe into json object and save json
        object into json file
        """
    data = json.load(js_file)
    df = pd.read_json(data, orient='records')
    print(df)
    """
    use json file to get the dataframe and print it out
    """
