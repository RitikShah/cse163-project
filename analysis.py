from time import time
import pandas as pd
import logging
import pickle
import sys

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

with open('model.pkl', 'rb') as file:
	