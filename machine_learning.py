from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


data = pd.read_pickle('data.pkl')
data = data.loc[:, data.columns != 'id']
x = data.loc[:, data.columns != 'readBy']
y = data['readBy']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


def ask_question(s):
    return str(input(s)).upper()[0] == 'Y'


if ask_question('test max_depth? [Y or N]: '):
    plot_data_1 = []

    # print('y_train', list(y_train)[0:3] + ['...'] + list(y_train)[-3:])
    # print('y_test', list(y_test)[0:3] + ['...'] + list(y_test)[-3:])

    for i in range(1, 100):
        model = DecisionTreeRegressor(max_depth=i)
        model.fit(x_train, y_train)

        train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)
        train_score = mean_squared_error(y_train, train_predict)
        test_score = mean_squared_error(y_test, test_predict)

        # print('#:', i)
        # print('train_predict', train_predict)
        # print('test_predict', test_predict)

        plot_data_1.append({'max depth': i, 'mean square error': train_score,
                            'predict type': 'trainning'})
        plot_data_1.append({'max depth': i, 'mean square error': test_score,
                            'predict type': 'testing'})

    plot_data_1 = pd.DataFrame(plot_data_1)
    test_data = plot_data_1[plot_data_1['predict type'] == 'testing']
    print(plot_data_1.loc[test_data['mean square error'].idxmin()])

    sns.relplot(kind='line', x='max depth', y='mean square error',
                hue='predict type', data=plot_data_1)
    plt.show()

if ask_question('test max_leaf_nodes? [Y or N]: '):
    plot_data_2 = []
    for i in range(2, 100):
        model = DecisionTreeRegressor(max_leaf_nodes=i)
        model.fit(x_train, y_train)

        train_score = mean_squared_error(y_train, model.predict(x_train))
        test_score = mean_squared_error(y_test, model.predict(x_test))

        plot_data_2.append({'max leaf nodes': i, 'mean square error':
                            train_score, 'predict type': 'trainning'})
        plot_data_2.append({'max leaf nodes': i, 'mean square error':
                            test_score, 'predict type': 'testing'})

    plot_data_2 = pd.DataFrame(plot_data_2)
    test_data = plot_data_2[plot_data_2['predict type'] == 'testing']
    print(plot_data_2.loc[test_data['mean square error'].idxmin()])

    sns.relplot(kind='line', x='max leaf nodes', y='mean square error',
                hue='predict type', data=plot_data_2)
    plt.show()

if ask_question('test min_samples_split? [Y or N]: '):
    plot_data_3 = []
    for i in range(2, 300):
        model = DecisionTreeRegressor(min_samples_split=i)
        model.fit(x_train, y_train)

        train_score = mean_squared_error(y_train, model.predict(x_train))
        test_score = mean_squared_error(y_test, model.predict(x_test))

        plot_data_3.append({'min sample split': i, 'mean square error':
                            train_score, 'predict type': 'trainning'})
        plot_data_3.append({'min sample split': i, 'mean square error':
                            test_score, 'predict type': 'testing'})

    plot_data_3 = pd.DataFrame(plot_data_3)
    test_data = plot_data_3[plot_data_3['predict type'] == 'testing']
    print(plot_data_3.loc[test_data['mean square error'].idxmin()])

    sns.relplot(kind='line', x='min sample split', y='mean square error',
                hue='predict type', data=plot_data_3)
    plt.show()

if ask_question('test min_samples_leaf? [Y or N]: '):
    plot_data_4 = []
    for i in range(1, 1000):
        model = DecisionTreeRegressor(min_samples_leaf=i)
        model.fit(x_train, y_train)

        train_score = mean_squared_error(y_train, model.predict(x_train))
        test_score = mean_squared_error(y_test, model.predict(x_test))

        plot_data_4.append({'min sample leaf': i, 'mean square error':
                            train_score, 'predict type': 'trainning'})
        plot_data_4.append({'min sample leaf': i, 'mean square error':
                            test_score, 'predict type': 'testing'})

    plot_data_4 = pd.DataFrame(plot_data_4)
    test_data = plot_data_4[plot_data_4['predict type'] == 'testing']
    print(plot_data_4.loc[test_data['mean square error'].idxmin()])

    sns.relplot(kind='line', x='min sample leaf', y='mean square error',
                hue='predict type', data=plot_data_4)
    plt.show()

if ask_question('test max_features? [Y or N]: '):
    plot_data_5 = []
    for i in range(1, 1000):
        model = DecisionTreeRegressor(min_samples_leaf=i)
        model.fit(x_train, y_train)

        train_score = mean_squared_error(y_train, model.predict(x_train))
        test_score = mean_squared_error(y_test, model.predict(x_test))

        plot_data_5.append({'max features': i, 'mean square error':
                            train_score, 'predict type': 'trainning'})
        plot_data_5.append({'max features': i, 'mean square error': test_score,
                            'predict type': 'testing'})

    plot_data_5 = pd.DataFrame(plot_data_5)
    test_data = plot_data_5[plot_data_5['predict type'] == 'testing']
    print(plot_data_5.loc[test_data['mean square error'].idxmin()])

    sns.relplot(kind='line', x='max features', y='mean square error',
                hue='predict type', data=plot_data_5)
    plt.show()
