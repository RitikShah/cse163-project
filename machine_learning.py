import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_pickle('data.pkl')
data = data.loc[:, data.columns != 'id']
data = data.replace([np.inf, -np.inf], np.nan)
data.dropna()
x = data.loc[:, data.columns != 'readBy']
y = data['readBy']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=1)


def ask_question(s):
    return str(input(s)).upper()[0] == 'Y'


if ask_question('test max_depth? [Y or N]: '):
    plot_data_1 = []
    for i in range(1, 100):
        model = DecisionTreeRegressor(max_depth=i, random_state=1)
        model.fit(x_train, y_train)

        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)

        plot_data_1.append({'max depth': i, 'train accuracy': train_score,
                            'test accuracy': test_score})

    plot_data_1 = pd.DataFrame(plot_data_1)
    print(plot_data_1.loc[plot_data_1['test accuracy'].idxmax()])

    fig, (ax1, ax2) = plt.subplots(2, figsize=(30, 30))
    sns.relplot(ax=ax1, kind='line', x='max depth', y='train accuracy',
                data=plot_data_1)
    sns.relplot(ax=ax2, kind='line', x='max depth', y='test accuracy',
                data=plot_data_1)
    plt.show()

elif ask_question('test max_leaf_nodes? [Y or N]: '):
    plot_data_2 = []
    for i in range(2, 100):
        model = DecisionTreeRegressor(max_leaf_nodes=i, random_state=1)
        model.fit(x_train, y_train)

        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)

        plot_data_2.append({'max leaf node': i, 'train accuracy': train_score,
                            'test accuracy': test_score})

    plot_data_2 = pd.DataFrame(plot_data_2)
    print(plot_data_2.loc[plot_data_2['test accuracy'].idxmax()])

    fig, (ax1, ax2) = plt.subplots(2, figsize=(30, 30))
    sns.relplot(ax=ax1, kind='line', x='max leaf node', y='train accuracy',
                data=plot_data_2)
    sns.relplot(ax=ax2, kind='line', x='max leaf node', y='test accuracy',
                data=plot_data_2)
    plt.show()

elif ask_question('test min_samples_split? [Y or N]: '):
    plot_data_3 = []
    for i in range(2, 100):
        model = DecisionTreeRegressor(min_samples_split=i, random_state=1)
        model.fit(x_train, y_train)

        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)

        plot_data_3.append({'min sample split': i, 'train accuracy':
                            train_score, 'test accuracy': test_score})

    plot_data_3 = pd.DataFrame(plot_data_3)
    print(plot_data_3.loc[plot_data_3['test accuracy'].idxmax()])

    fig, (ax1, ax2) = plt.subplots(2, figsize=(30, 30))
    sns.relplot(ax=ax1, kind='line', x='min sample split', y='train accuracy',
                data=plot_data_3)
    sns.relplot(ax=ax2, kind='line', x='min sample split', y='test accuracy',
                data=plot_data_3)
    plt.show()
