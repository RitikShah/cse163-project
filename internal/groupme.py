from .clean_data import clean_sentence
from .features import get_features
from .utils import remove_col

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DATA_FILE = 'groupme/message.json'


def main():
    """ Prepares the groupme data """
    logger.info('loading in json') 
    with open(DATA_FILE) as file:
        data = pd.read_json(file)[['user_id', 'name', 'text',
                                   'favorited_by', 'id',
                                   'attachments']]
    data['text'] = data['text'].fillna(value='')
    data = data.fillna(value=0)

    logger.info('calculating likes and attachments')
    data['readBy'] = data['favorited_by'].apply(len)
    data['attachments'] = data['attachments'].apply(len)
    data['fromUser.id'] = data['user_id']
    data['text_clean'] = data['text'].apply(clean_sentence)

    data['mentions'] = data['text'].str.count('@')
    data['urls'] = 0

    data = remove_col(data, 'user_id')
    data = remove_col(data, 'favorited_by')

    data = get_features(data)
    logger.info('saving to groupme')
    data.to_pickle('pickles/groupme.pkl')


if __name__ == '__main__':
    main()
