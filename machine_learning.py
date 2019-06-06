from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
# TODO: throw this in a function pls

# hyper parameters constants
DEPTH = 5
SAMPLES_SPLIT = 0.13
SAMPLES_LEAF = 0.02
WEIGHT_FRACTION_LEAF = 0.02
LEAF_NODES = 29
IMPURITY_DECREASE = 0.03

logging.info('unpickling')
data = pd.read_pickle('featured.pkl')
logging.info('dividing columns')
ml_data = data = data.loc[:, data.columns != 'id']
x = ml_data.loc[:, ml_data.columns != 'readBy']
y = ml_data['readBy']
logging.info('splitting data: test/train')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

logging.info('creating regressor')
# hyper parameters were tested in an earlier program
model = DecisionTreeRegressor(
    max_depth=DEPTH,
    min_samples_split=SAMPLES_SPLIT,
    min_samples_leaf=SAMPLES_LEAF,
    min_weight_fraction_leaf=WEIGHT_FRACTION_LEAF,
    max_leaf_nodes=LEAF_NODES,
    min_impurity_decrease=IMPURITY_DECREASE
)

logging.info('fitting model')
model.fit(x_train, y_train)

logging.info('making predictions about test set')
a = list(model.predict(x_test))
b = list(y_test)

logging.info('calculating error')
close = 0
for i in range(len(a)):
    print(b[i], a[i])
    if (a[i] >= (b[i] - 1)) & (a[i] <= (b[i] + 1)):
        close += 1
print(mean_squared_error(y_test, model.predict(x_test)))
print('out of', str(len(b)), str(close), 'predict is close')

# df = pd.DataFrame([y_train, model.predict(x_test)])
# print(df)
