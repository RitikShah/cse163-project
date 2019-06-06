# from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from split import train, dev, test
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def ask_question(s):
    return str(input(s)).upper()[0] == 'Y'


def remove_id(dset):
    return dset.loc[:, dset.columns != 'id']


def x_y(dset):
    return dset.loc[:, dset.columns != 'readBy'], dset['readBy']


def isolated_test(train, dev, test):
    train = remove_id(train)
    dev = remove_id(dev)
    test = remove_id(test)

    x_train, y_train = x_y(train)
    x_dev, y_dev = x_y(dev)
    x_test, y_test = x_y(test)

    if ask_question('test max_depth? [Y or N]: '):
        min_mse = None
        min_model = None
        min_depth = 0

        # print('y_train', list(y_train)[0:3] + ['...'] + list(y_train)[-3:])
        # print('y_test', list(y_test)[0:3] + ['...'] + list(y_test)[-3:])

        for i in range(1, 100, 10):
            model = DecisionTreeRegressor(max_depth=i)
            model.fit(x_train, y_train)

            train_predict = model.predict(x_train)
            dev_predict = model.predict(x_test)
            train_score = mean_squared_error(y_train, train_predict)
            dev_score = mean_squared_error(y_dev, dev_predict)

            # print('#:', i)
            # print('train_predict', train_predict)
            # print('test_predict', test_predict)

            print({'max depth': i,
                   'mse train': train_score,
                   'mse test': dev_score})

            if min_mse is None or dev_score < min_mse:
                min_mse = dev_score
                min_depth = i
                min_model = model

        # plot_data_1 = pd.DataFrame(plot_data_1)
        # test_data = plot_data_1[plot_data_1['predict type'] == 'testing']
        # print(plot_data_1.loc[test_data['mean square error'].idxmin()])
        print(f'dev mse: {min_mse}')
        print(f'depth: {min_depth}')

        test_score = mean_squared_error(y_test, min_model.predict(x_test))
        print(f'test mse: {test_score}')


        # sns.relplot(kind='line', x='max depth', y='mean square error',
        #  hue='predict type', data=plot_data_1)
        # plt.savefig('max_depth.png')

    if ask_question('test min_samples_split? [Y or N]: '):
        min_mse = None
        min_model = None
        min_depth = 0

        for i in range(2, 100, 10):
            model = DecisionTreeRegressor(min_samples_split=i)
            model.fit(x_train, y_train)

            train_predict = model.predict(x_train)
            dev_predict = model.predict(x_test)
            train_score = mean_squared_error(y_train, train_predict)
            dev_score = mean_squared_error(y_dev, dev_predict)

            print({'max depth': i,
                   'mse train': train_score,
                   'mse test': dev_score})

            if min_mse is None or dev_score < min_mse:
                min_mse = dev_score
                min_depth = i
                min_model = model

        print(f'dev mse: {min_mse}')
        print(f'depth: {min_depth}')

        test_score = mean_squared_error(y_test, min_model.predict(x_test))
        print(f'test mse: {test_score}')

    if ask_question('test min_samples_leaf? [Y or N]: '):
        min_mse = None
        min_model = None
        min_depth = 0

        for i in range(1, 51, 10):
            i = i / 100
            model = DecisionTreeRegressor(min_samples_leaf=i)
            model.fit(x_train, y_train)

            train_predict = model.predict(x_train)
            dev_predict = model.predict(x_test)
            train_score = mean_squared_error(y_train, train_predict)
            dev_score = mean_squared_error(y_dev, dev_predict)

            print({'max depth': i,
                   'mse train': train_score,
                   'mse test': dev_score})

            if min_mse is None or dev_score < min_mse:
                min_mse = dev_score
                min_depth = i
                min_model = model

        print(f'dev mse: {min_mse}')
        print(f'depth: {min_depth}')

        test_score = mean_squared_error(y_test, min_model.predict(x_test))
        print(f'test mse: {test_score}')

    if ask_question('test min_weight_fraction_leaf ? [Y or N]: '):
        min_mse = None
        min_model = None
        min_depth = 0

        for i in range(1, 51, 10):
            i = i / 10
            model = DecisionTreeRegressor(min_weight_fraction_leaf=i)
            model.fit(x_train, y_train)

            train_predict = model.predict(x_train)
            dev_predict = model.predict(x_test)
            train_score = mean_squared_error(y_train, train_predict)
            dev_score = mean_squared_error(y_dev, dev_predict)

            print({'max depth': i,
                   'mse train': train_score,
                   'mse test': dev_score})

            if min_mse is None or dev_score < min_mse:
                min_mse = dev_score
                min_depth = i
                min_model = model

        print(f'dev mse: {min_mse}')
        print(f'depth: {min_depth}')

        test_score = mean_squared_error(y_test, min_model.predict(x_test))
        print(f'test mse: {test_score}')

    if ask_question('test max_features? [Y or N]: '):
        min_mse = None
        min_model = None
        min_depth = 0

        for i in range(1, 51, 10):
            i = i / 10
            model = DecisionTreeRegressor(max_features=i)
            model.fit(x_train, y_train)

            train_predict = model.predict(x_train)
            dev_predict = model.predict(x_test)
            train_score = mean_squared_error(y_train, train_predict)
            dev_score = mean_squared_error(y_dev, dev_predict)

            print({'max depth': i,
                   'mse train': train_score,
                   'mse test': dev_score})

            if min_mse is None or dev_score < min_mse:
                min_mse = dev_score
                min_depth = i
                min_model = model

        print(f'dev mse: {min_mse}')
        print(f'depth: {min_depth}')

        test_score = mean_squared_error(y_test, min_model.predict(x_test))
        print(f'test mse: {test_score}')

    if ask_question('test max_leaf_nodes? [Y or N]: '):
        min_mse = None
        min_model = None
        min_depth = 0

        for i in range(2, 100, 10):
            model = DecisionTreeRegressor(max_leaf_nodest=i)
            model.fit(x_train, y_train)

            train_predict = model.predict(x_train)
            dev_predict = model.predict(x_test)
            train_score = mean_squared_error(y_train, train_predict)
            dev_score = mean_squared_error(y_dev, dev_predict)

            print({'max depth': i,
                   'mse train': train_score,
                   'mse test': dev_score})

            if min_mse is None or dev_score < min_mse:
                min_mse = dev_score
                min_depth = i
                min_model = model

        print(f'dev mse: {min_mse}')
        print(f'depth: {min_depth}')

        test_score = mean_squared_error(y_test, min_model.predict(x_test))
        print(f'test mse: {test_score}')

    if ask_question('test min_impurity_decrease? [Y or N]: '):
        min_mse = None
        min_model = None
        min_depth = 0

        for i in range(1, 100, 10):
            model = DecisionTreeRegressor(min_impurity_decrease=i)
            model.fit(x_train, y_train)

            train_predict = model.predict(x_train)
            dev_predict = model.predict(x_test)
            train_score = mean_squared_error(y_train, train_predict)
            dev_score = mean_squared_error(y_dev, dev_predict)

            print({'max depth': i,
                   'mse train': train_score,
                   'mse test': dev_score})

            if min_mse is None or dev_score < min_mse:
                min_mse = dev_score
                min_depth = i
                min_model = model

        print(f'dev mse: {min_mse}')
        print(f'depth: {min_depth}')

        test_score = mean_squared_error(y_test, min_model.predict(x_test))
        print(f'test mse: {test_score}')

    if ask_question('test min_impurity_split? [Y or N]: '):
        min_mse = None
        min_model = None
        min_depth = 0

        for i in range(1, 100, 10):
            model = DecisionTreeRegressor(min_impurity_split=i)
            model.fit(x_train, y_train)

            train_predict = model.predict(x_train)
            dev_predict = model.predict(x_test)
            train_score = mean_squared_error(y_train, train_predict)
            dev_score = mean_squared_error(y_dev, dev_predict)

            print({'max depth': i,
                   'mse train': train_score,
                   'mse test': dev_score})

            if min_mse is None or dev_score < min_mse:
                min_mse = dev_score
                min_depth = i
                min_model = model

        print(f'dev mse: {min_mse}')
        print(f'depth: {min_depth}')

        test_score = mean_squared_error(y_test, min_model.predict(x_test))
        print(f'test mse: {test_score}')


def focused_test(pickle):
    data = pd.read_pickle(pickle)
    data = data.loc[:, data.columns != 'id']
    x = data.loc[:, data.columns != 'readBy']
    y = data['readBy']
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40)
    # for i in range()


def main():
    isolated_test(train(), dev(), test())


if __name__ == "__main__":
    main()
