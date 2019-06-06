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
    plt.savefig('max_depth.png')

if ask_question('test min_samples_split? [Y or N]: '):
    plot_data_2 = []
    for i in range(1, 100):
        i = i / 100
        model = DecisionTreeRegressor(min_samples_split=i)
        model.fit(x_train, y_train)

        train_score = mean_squared_error(y_train, model.predict(x_train))
        test_score = mean_squared_error(y_test, model.predict(x_test))

        plot_data_2.append({'min samples split': i, 'mean square error':
                            train_score, 'predict type': 'trainning'})
        plot_data_2.append({'min samples split': i, 'mean square error':
                            test_score, 'predict type': 'testing'})

    plot_data_2 = pd.DataFrame(plot_data_2)
    test_data = plot_data_2[plot_data_2['predict type'] == 'testing']
    print(plot_data_2.loc[test_data['mean square error'].idxmin()])

    sns.relplot(kind='line', x='min samples split', y='mean square error',
                hue='predict type', data=plot_data_2)
    plt.savefig('min_samples_split.png')

if ask_question('test min_samples_leaf? [Y or N]: '):
    plot_data_3 = []
    for i in range(1, 51):
        i = i / 100
        model = DecisionTreeRegressor(min_samples_leaf=i)
        model.fit(x_train, y_train)

        train_score = mean_squared_error(y_train, model.predict(x_train))
        test_score = mean_squared_error(y_test, model.predict(x_test))

        plot_data_3.append({'min sample leaf': i, 'mean square error':
                            train_score, 'predict type': 'trainning'})
        plot_data_3.append({'min sample leaf': i, 'mean square error':
                            test_score, 'predict type': 'testing'})

    plot_data_3 = pd.DataFrame(plot_data_3)
    test_data = plot_data_3[plot_data_3['predict type'] == 'testing']
    print(plot_data_3.loc[test_data['mean square error'].idxmin()])

    sns.relplot(kind='line', x='min sample leaf', y='mean square error',
                hue='predict type', data=plot_data_3)
    plt.savefig('min_samples_leaf.png')

if ask_question('test min_weight_fraction_leaf ? [Y or N]: '):
    plot_data_4 = []
    for i in range(1, 51):
        i = i / 100
        model = DecisionTreeRegressor(min_weight_fraction_leaf=i)
        model.fit(x_train, y_train)

        train_score = mean_squared_error(y_train, model.predict(x_train))
        test_score = mean_squared_error(y_test, model.predict(x_test))

        plot_data_4.append({'min weight fraction leaf': i, 'mean square error':
                            train_score, 'predict type': 'trainning'})
        plot_data_4.append({'min weight fraction leaf': i, 'mean square error':
                            test_score, 'predict type': 'testing'})

    plot_data_4 = pd.DataFrame(plot_data_4)
    test_data = plot_data_4[plot_data_4['predict type'] == 'testing']
    print(plot_data_4.loc[test_data['mean square error'].idxmin()])

    sns.relplot(kind='line', x='min weight fraction leaf',
                y='mean square error', hue='predict type', data=plot_data_4)
    plt.savefig('min_weight_fraction_leaf.png')

if ask_question('test max_features? [Y or N]: '):
    plot_data_5 = []
    for i in range(1, 51):
        i = i / 100
        model = DecisionTreeRegressor(max_features=i)
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
    plt.savefig('max_features.png')

if ask_question('test max_leaf_nodes? [Y or N]: '):
    plot_data_6 = []
    for i in range(2, 100):
        model = DecisionTreeRegressor(max_leaf_nodes=i)
        model.fit(x_train, y_train)

        train_score = mean_squared_error(y_train, model.predict(x_train))
        test_score = mean_squared_error(y_test, model.predict(x_test))

        plot_data_6.append({'max leaf nodes': i, 'mean square error':
                            train_score, 'predict type': 'trainning'})
        plot_data_6.append({'max leaf nodes': i, 'mean square error':
                            test_score, 'predict type': 'testing'})

    plot_data_6 = pd.DataFrame(plot_data_6)
    test_data = plot_data_6[plot_data_6['predict type'] == 'testing']
    print(plot_data_6.loc[test_data['mean square error'].idxmin()])

    sns.relplot(kind='line', x='max leaf nodes', y='mean square error',
                hue='predict type', data=plot_data_6)
    plt.savefig('max_leaf_nodes.png')

if ask_question('test min_impurity_decrease? [Y or N]: '):
    plot_data_7 = []
    for i in range(1, 100):
        i = i / 100
        model = DecisionTreeRegressor(min_impurity_decrease=i)
        model.fit(x_train, y_train)

        train_score = mean_squared_error(y_train, model.predict(x_train))
        test_score = mean_squared_error(y_test, model.predict(x_test))

        plot_data_7.append({'min impurity decrease': i, 'mean square error':
                            train_score, 'predict type': 'trainning'})
        plot_data_7.append({'min impurity decrease': i, 'mean square error':
                            test_score, 'predict type': 'testing'})

    plot_data_7 = pd.DataFrame(plot_data_7)
    test_data = plot_data_7[plot_data_7['predict type'] == 'testing']
    print(plot_data_7.loc[test_data['mean square error'].idxmin()])

    sns.relplot(kind='line', x='min impurity decrease', y='mean square error',
                hue='predict type', data=plot_data_7)
    plt.savefig('min_inpurity_decrease.png')

if ask_question('test min_impurity_split? [Y or N]: '):
    plot_data_8 = []
    for i in range(1, 100):
        i = i / 100
        model = DecisionTreeRegressor(min_impurity_split=i)
        model.fit(x_train, y_train)

        train_score = mean_squared_error(y_train, model.predict(x_train))
        test_score = mean_squared_error(y_test, model.predict(x_test))

        plot_data_8.append({'min impurity split': i, 'mean square error':
                            train_score, 'predict type': 'trainning'})
        plot_data_8.append({'min impurity split': i, 'mean square error':
                            test_score, 'predict type': 'testing'})

    plot_data_8 = pd.DataFrame(plot_data_8)
    test_data = plot_data_8[plot_data_8['predict type'] == 'testing']
    print(plot_data_8.loc[test_data['mean square error'].idxmin()])

    sns.relplot(kind='line', x='min impurity split', y='mean square error',
                hue='predict type', data=plot_data_8)
    plt.savefig('min_impurity_split.png')
