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

        plot_data_1.append({'max depth': i, 'train accuracy': train_score,
                            'test accuracy': test_score})

    plot_data_1 = pd.DataFrame(plot_data_1)
    print(plot_data_1.loc[plot_data_1['test accuracy'].idxmin()])

    fig, ax = plt.subplots(1, figsize=(30, 30))
    sns.relplot(ax=ax, kind='line', x='max depth', y='train accuracy',
                color='red', data=plot_data_1)
    sns.relplot(ax=ax, kind='line', x='max depth', y='test accuracy',
                data=plot_data_1)
    plt.show()

if ask_question('test max_leaf_nodes? [Y or N]: '):
    plot_data_2 = []
    for i in range(2, 100):
        model = DecisionTreeRegressor(max_leaf_nodes=i)
        model.fit(x_train, y_train)

        train_score = mean_squared_error(y_train, model.predict(x_train))
        test_score = mean_squared_error(y_test, model.predict(x_test))

        plot_data_2.append({'max leaf node': i, 'train accuracy': train_score,
                            'test accuracy': test_score})

    plot_data_2 = pd.DataFrame(plot_data_2)
    print(plot_data_2.loc[plot_data_2['test accuracy'].idxmin()])

    fig, (ax1, ax2) = plt.subplots(2, figsize=(30, 30))
    sns.relplot(ax=ax1, kind='line', x='max leaf node', y='train accuracy',
                data=plot_data_2)
    sns.relplot(ax=ax2, kind='line', x='max leaf node', y='test accuracy',
                data=plot_data_2)
    plt.show()

if ask_question('test min_samples_split? [Y or N]: '):
    plot_data_3 = []
    for i in range(2, 200):
        model = DecisionTreeRegressor(min_samples_split=i)
        model.fit(x_train, y_train)

        train_score = mean_squared_error(y_train, model.predict(x_train))
        test_score = mean_squared_error(y_test, model.predict(x_test))

        plot_data_3.append({'min sample split': i, 'train accuracy':
                            train_score, 'test accuracy': test_score})

    plot_data_3 = pd.DataFrame(plot_data_3)
    print(plot_data_3.loc[plot_data_3['test accuracy'].idxmin()])

    fig, (ax1, ax2) = plt.subplots(2, figsize=(30, 30))
    sns.relplot(ax=ax1, kind='line', x='min sample split', y='train accuracy',
                data=plot_data_3)
    sns.relplot(ax=ax2, kind='line', x='min sample split', y='test accuracy',
                data=plot_data_3)
    plt.show()

if ask_question('test min_samples_leaf? [Y or N]: '):
    plot_data_4 = []
    for i in range(1, 200):
        model = DecisionTreeRegressor(min_samples_leaf=i)
        model.fit(x_train, y_train)

        train_score = mean_squared_error(y_train, model.predict(x_train))
        test_score = mean_squared_error(y_test, model.predict(x_test))

        plot_data_4.append({'min sample leaf': i, 'train accuracy':
                            train_score, 'test accuracy': test_score})

    plot_data_4 = pd.DataFrame(plot_data_4)
    print(plot_data_4.loc[plot_data_4['test accuracy'].idxmin()])

    fig, (ax1, ax2) = plt.subplots(2, figsize=(30, 30))
    sns.relplot(ax=ax1, kind='line', x='min sample leaf', y='train accuracy',
                data=plot_data_4)
    sns.relplot(ax=ax2, kind='line', x='min sample leaf', y='test accuracy',
                data=plot_data_4)
    plt.show()

if ask_question('test max_features? [Y or N]: '):
    plot_data_5 = []
    for i in range(1, 1000):
        model = DecisionTreeRegressor(min_samples_leaf=i)
        model.fit(x_train, y_train)

        train_score = mean_squared_error(y_train, model.predict(x_train))
        test_score = mean_squared_error(y_test, model.predict(x_test))

        plot_data_5.append({'max features': i, 'train accuracy':
                            train_score, 'test accuracy': test_score})

    plot_data_5 = pd.DataFrame(plot_data_5)
    print(plot_data_5.loc[plot_data_5['test accuracy'].idxmin()])

    fig, (ax1, ax2) = plt.subplots(2, figsize=(30, 30))
    sns.relplot(ax=ax1, kind='line', x='max features', y='train accuracy',
                data=plot_data_5)
    sns.relplot(ax=ax2, kind='line', x='max features', y='test accuracy',
                data=plot_data_5)
    plt.show()
