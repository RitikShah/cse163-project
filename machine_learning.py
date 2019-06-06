from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

# TODO: throw this in a function pls

# hyper parameters constants
DEPTH = 6
SAMPLES_SPLIT = 0.04
SAMPLES_LEAF = 0.01
WEIGHT_FRACTION_LEAF = 0.01
LEAF_NODES = 42
FEATURES = 0.26
IMPURITY_DECREASE = 0.01

data = pd.read_pickle('featured.pkl')
ml_data = data = data.loc[:, data.columns != 'id']
x = ml_data.loc[:, ml_data.columns != 'readBy']
y = ml_data['readBy']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# hyper parameters were tested in an earlier program
model = DecisionTreeRegressor(
    max_depth=DEPTH,
    min_samples_split=SAMPLES_SPLIT,
    min_samples_leaf=SAMPLES_LEAF,
    min_weight_fraction_leaf=WEIGHT_FRACTION_LEAF,
    max_leaf_nodes=LEAF_NODES,
    min_impurity_decrease=IMPURITY_DECREASE,
    max_features=FEATURES
)
model.fit(x_train, y_train)

a = list(model.predict(x_test))
b = list(y_test)

close = 0
for i in range(len(a)):
    print(b[i], a[i])
    if (a[i] >= (b[i] - 1)) & (a[i] <= (b[i] + 1)):
        close += 1
print(mean_squared_error(y_test, model.predict(x_test)))
print('out of', str(len(b)), str(close), 'predict is close')

# df = pd.DataFrame([y_train, model.predict(x_test)])
# print(df)
