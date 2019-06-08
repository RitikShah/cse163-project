from .clean_data import clean_sentence

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DATA_FILE = 'groupme/message.json'


def main():
    logger.info('loading in json')
    with open(DATA_FILE) as file:
        data = pd.read_json(file)[['user_id', 'name', 'text',
                                   'favorited_by',
                                   'attachments']]
    data['text'] = data['text'].fillna(value='')
    data = data.fillna(value=0)

    logger.info('calculating likes and attachments')
    data['likes'] = data['favorited_by'].apply(len)
    data['attachments'] = data['attachments'].apply(len)
    data = data[['user_id', 'likes', 'attachments', 'text', 'name']]
    data['text_clean'] = data['text'].apply(clean_sentence)

    logger.info('pickling to cleaned.pkl')
    data.to_pickle('cleaned.pkl')


if __name__ == '__main__':
    main()
