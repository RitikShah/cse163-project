import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


data = pd.read_pickle('data.pkl')
ml_data = data = data.loc[:, data.columns != 'id']
x = ml_data.loc[:, ml_data.columns != 'readBy']
y = ml_data['readBy']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = DecisionTreeRegressor(max_depth=5, min_samples_split=0.07,
                              min_samples_leaf=0.09,
                              min_weight_fraction_leaf=0.09,
                              max_leaf_nodes=21, min_impurity_decrease=0.01)
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
