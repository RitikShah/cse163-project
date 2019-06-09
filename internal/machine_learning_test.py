from .utils import remove_col, x_y, ask_question
from .split import get_train, get_dev, get_test

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def graph_analysis(train, dev):
    """ This tests 8 parameters and graphs over accuracy """
    train = remove_col(remove_col(train, 'fromUser.id'), 'id')
    dev = remove_col(remove_col(dev, 'fromUser.id'), 'id')
    train_frac = dev.sample(frac=0.05)
    dev_frac = dev.sample(frac=0.05)
    x_train, y_train = x_y(train_frac, 'readBy')
    x_dev, y_dev = x_y(dev_frac, 'readBy')

    if ask_question('test max_depth? [Y or N]: '):
        plot_data_1 = []
        for i in range(1, 100):
            model = DecisionTreeRegressor(max_depth=i)
            model.fit(x_train, y_train)

            train_score = mean_squared_error(y_train, model.predict(x_train))
            test_score = mean_squared_error(y_dev, model.predict(x_dev))

            plot_data_1.append({'min samples split': i, 'mean square error':
                                train_score, 'predict type': 'trainning'})
            plot_data_1.append({'min samples split': i, 'mean square error':
                                test_score, 'predict type': 'testing'})

        plot_data_1 = pd.DataFrame(plot_data_1)
        test_data = plot_data_1[plot_data_1['predict type'] == 'testing']
        print(plot_data_1.loc[test_data['mean square error'].idxmin()])

        sns.relplot(kind='line', x='min samples split', y='mean square error',
                    hue='predict type', data=plot_data_1)
        plt.savefig('graphs/min_samples_split.png')

    if ask_question('test min_samples_split? [Y or N]: '):
        plot_data_2 = []
        for i in range(1, 100):
            i = i / 100
            model = DecisionTreeRegressor(min_samples_split=i)
            model.fit(x_train, y_train)

            train_score = mean_squared_error(y_train, model.predict(x_train))
            test_score = mean_squared_error(y_dev, model.predict(x_dev))

            plot_data_2.append({'min samples split': i, 'mean square error':
                                train_score, 'predict type': 'training'})
            plot_data_2.append({'min samples split': i, 'mean square error':
                                test_score, 'predict type': 'testing'})

        plot_data_2 = pd.DataFrame(plot_data_2)
        test_data = plot_data_2[plot_data_2['predict type'] == 'testing']
        print(plot_data_2.loc[test_data['mean square error'].idxmin()])

        sns.relplot(kind='line', x='min samples split', y='mean square error',
                    hue='predict type', data=plot_data_2)
        plt.savefig('graphs/min_samples_split.png')

    if ask_question('test min_samples_leaf? [Y or N]: '):
        plot_data_3 = []
        for i in range(1, 51):
            i = i / 100
            model = DecisionTreeRegressor(min_samples_leaf=i)
            model.fit(x_train, y_train)

            train_score = mean_squared_error(y_train, model.predict(x_train))
            test_score = mean_squared_error(y_dev, model.predict(x_dev))

            plot_data_3.append({'min sample leaf': i, 'mean square error':
                                train_score, 'predict type': 'training'})
            plot_data_3.append({'min sample leaf': i, 'mean square error':
                                test_score, 'predict type': 'testing'})

        plot_data_3 = pd.DataFrame(plot_data_3)
        test_data = plot_data_3[plot_data_3['predict type'] == 'testing']
        print(plot_data_3.loc[test_data['mean square error'].idxmin()])

        sns.relplot(kind='line', x='min sample leaf', y='mean square error',
                    hue='predict type', data=plot_data_3)
        plt.savefig('graphs/min_samples_leaf.png')

    if ask_question('test min_weight_fraction_leaf ? [Y or N]: '):
        plot_data_4 = []
        for i in range(1, 51):
            i = i / 100
            model = DecisionTreeRegressor(min_weight_fraction_leaf=i)
            model.fit(x_train, y_train)

            train_score = mean_squared_error(y_train, model.predict(x_train))
            test_score = mean_squared_error(y_dev, model.predict(x_dev))

            plot_data_4.append({'min weight fraction leaf': i, 'mean square error':  # noqa
                                train_score, 'predict type': 'training'})
            plot_data_4.append({'min weight fraction leaf': i, 'mean square error':  # noqa
                                test_score, 'predict type': 'testing'})

        plot_data_4 = pd.DataFrame(plot_data_4)
        test_data = plot_data_4[plot_data_4['predict type'] == 'testing']
        print(plot_data_4.loc[test_data['mean square error'].idxmin()])

        sns.relplot(kind='line', x='min weight fraction leaf',
                    y='mean square error', hue='predict type', data=plot_data_4)  # noqa
        plt.savefig('graphs/min_weight_fraction_leaf.png')

    if ask_question('test max_features? [Y or N]: '):
        plot_data_5 = []
        for i in range(1, 51):
            i = i / 100
            model = DecisionTreeRegressor(max_features=i)
            model.fit(x_train, y_train)

            train_score = mean_squared_error(y_train, model.predict(x_train))
            test_score = mean_squared_error(y_dev, model.predict(x_dev))

            plot_data_5.append({'max features': i, 'mean square error':
                                train_score, 'predict type': 'training'})
            plot_data_5.append({'max features': i, 'mean square error': test_score,  # noqa
                                'predict type': 'testing'})

        plot_data_5 = pd.DataFrame(plot_data_5)
        test_data = plot_data_5[plot_data_5['predict type'] == 'testing']
        print(plot_data_5.loc[test_data['mean square error'].idxmin()])

        sns.relplot(kind='line', x='max features', y='mean square error',
                    hue='predict type', data=plot_data_5)
        plt.savefig('graphs/max_features.png')

    if ask_question('test max_leaf_nodes? [Y or N]: '):
        plot_data_6 = []
        for i in range(2, 100):
            model = DecisionTreeRegressor(max_leaf_nodes=i)
            model.fit(x_train, y_train)

            train_score = mean_squared_error(y_train, model.predict(x_train))
            test_score = mean_squared_error(y_dev, model.predict(x_dev))

            plot_data_6.append({'max leaf nodes': i, 'mean square error':
                                train_score, 'predict type': 'training'})
            plot_data_6.append({'max leaf nodes': i, 'mean square error':
                                test_score, 'predict type': 'testing'})

        plot_data_6 = pd.DataFrame(plot_data_6)
        test_data = plot_data_6[plot_data_6['predict type'] == 'testing']
        print(plot_data_6.loc[test_data['mean square error'].idxmin()])

        sns.relplot(kind='line', x='max leaf nodes', y='mean square error',
                    hue='predict type', data=plot_data_6)
        plt.savefig('graphs/max_leaf_nodes.png')

    if ask_question('test min_impurity_decrease? [Y or N]: '):
        plot_data_7 = []
        for i in range(1, 100):
            i = i / 100
            model = DecisionTreeRegressor(min_impurity_decrease=i)
            model.fit(x_train, y_train)

            train_score = mean_squared_error(y_train, model.predict(x_train))
            test_score = mean_squared_error(y_dev, model.predict(x_dev))

            plot_data_7.append({'min impurity decrease': i, 'mean square error':  # noqa
                                train_score, 'predict type': 'training'})
            plot_data_7.append({'min impurity decrease': i, 'mean square error':  # noqa
                                test_score, 'predict type': 'testing'})

        plot_data_7 = pd.DataFrame(plot_data_7)
        test_data = plot_data_7[plot_data_7['predict type'] == 'testing']
        print(plot_data_7.loc[test_data['mean square error'].idxmin()])

        sns.relplot(kind='line', x='min impurity decrease', y='mean square error',  # noqa
                    hue='predict type', data=plot_data_7)
        plt.savefig('graphs/min_inpurity_decrease.png')


def depth(x_train, y_train, x_dev, y_dev, x_test, y_test):
    """ isolated depth tests """
    min_mse = None
    min_model = None
    min_depth = 0

    logger.info('testing max_depth with max_leaf_nodes=37')
    for i in range(1, 21):
        model = DecisionTreeRegressor(max_depth=i, max_leaf_nodes=37)
        model.fit(x_train, y_train)

        train_predict = model.predict(x_train)
        dev_predict = model.predict(x_dev)
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
    print(f'max_depth: {min_depth}')

    test_score = mean_squared_error(y_test, min_model.predict(x_test))
    print(f'test mse: {test_score}')


def samples_split(x_train, y_train, x_dev, y_dev, x_test, y_test):
    """ isolated samples split tests """
    min_mse = None
    min_model = None
    min_depth = 0

    logger.info('testing min_samples_split')
    for i in range(2, 100, 10):
        model = DecisionTreeRegressor(min_samples_split=i)
        model.fit(x_train, y_train)

        train_predict = model.predict(x_train)
        dev_predict = model.predict(x_dev)
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
    print(f'min_samples_split: {min_depth}')

    test_score = mean_squared_error(y_test, min_model.predict(x_test))
    print(f'test mse: {test_score}')


def samples_leaf(x_train, y_train, x_dev, y_dev, x_test, y_test):
    """ isolated samples leaf tests """
    min_mse = None
    min_model = None
    min_depth = 0

    logger.info('testing min_samples_leaf')
    for i in range(1, 51, 10):
        i = i / 100
        model = DecisionTreeRegressor(min_samples_leaf=i)
        model.fit(x_train, y_train)

        train_predict = model.predict(x_train)
        dev_predict = model.predict(x_dev)
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
    print(f'min_samples_leaf: {min_depth}')

    test_score = mean_squared_error(y_test, min_model.predict(x_test))
    print(f'test mse: {test_score}')


def fraction_leaf(x_train, y_train, x_dev, y_dev, x_test, y_test):
    """ isolated fraction leafs tests """
    min_mse = None
    min_model = None
    min_depth = 0

    logger.info('testing min_weight_fraction_leaf')
    for i in range(1, 51, 10):
        i = i / 10
        model = DecisionTreeRegressor(min_weight_fraction_leaf=i)
        model.fit(x_train, y_train)

        train_predict = model.predict(x_train)
        dev_predict = model.predict(x_dev)
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
    print(f'min_weight_fraction_leaf: {min_depth}')

    test_score = mean_squared_error(y_test, min_model.predict(x_test))
    print(f'test mse: {test_score}')


def features(x_train, y_train, x_dev, y_dev, x_test, y_test):
    """ isolated features tests """
    min_mse = None
    min_model = None
    min_depth = 0

    logger.info('testing max_features')
    for i in range(1, 51, 10):
        i = i / 10
        model = DecisionTreeRegressor(max_features=i)
        model.fit(x_train, y_train)

        train_predict = model.predict(x_train)
        dev_predict = model.predict(x_dev)
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
    print(f'max_features: {min_depth}')

    test_score = mean_squared_error(y_test, min_model.predict(x_test))
    print(f'test mse: {test_score}')


def leaf_nodes(x_train, y_train, x_dev, y_dev, x_test, y_test):
    """ isolated lead nodes """
    min_mse = None
    min_model = None
    min_depth = 0

    logger.info('testing max_leaf_nodes with max_depth=7')
    for i in range(22, 43):
        model = DecisionTreeRegressor(max_leaf_nodes=i, max_depth=7)
        model.fit(x_train, y_train)

        train_predict = model.predict(x_train)
        dev_predict = model.predict(x_dev)
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
    print(f'max_leaf_nodes: {min_depth}')

    test_score = mean_squared_error(y_test, min_model.predict(x_test))
    print(f'test mse: {test_score}')


def impurity_decrease(x_train, y_train, x_dev, y_dev, x_test, y_test):
    """ isolated impurity decerease tests """
    min_mse = None
    min_model = None
    min_depth = 0

    logger.info('testing min_impurity_decrease')
    for i in range(1, 25):
        model = DecisionTreeRegressor(min_impurity_decrease=i)
        model.fit(x_train, y_train)

        train_predict = model.predict(x_train)
        dev_predict = model.predict(x_dev)
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
    print(f'min_impurity_decrease: {min_depth}')

    test_score = mean_squared_error(y_test, min_model.predict(x_test))
    print(f'test mse: {test_score}')


def isolated_test(train, dev, test):
    """ isolated tests """
    if ask_question('do isolated test? [Y or N]: '):
        logger.info('focused test 4')
        train = remove_col(remove_col(train, 'fromUser.id'), 'id')
        dev = remove_col(remove_col(dev, 'fromUser.id'), 'id')
        test = remove_col(remove_col(test, 'fromUser.id'), 'id')

        x_train, y_train = x_y(train, 'readBy')
        x_dev, y_dev = x_y(dev, 'readBy')
        x_test, y_test = x_y(test, 'readBy')

        depth(x_train, y_train, x_dev, y_dev, x_test, y_test)
        leaf_nodes(x_train, y_train, x_dev, y_dev, x_test, y_test)
        impurity_decrease(x_train, y_train, x_dev, y_dev, x_test, y_test)


def focused_test(train, dev, test):
    """ focused tests """
    if ask_question('do focused test? [Y or N]: '):
        logger.info('focused test 5')
        train = remove_col(remove_col(train, 'fromUser.id'), 'id')
        dev = remove_col(remove_col(dev, 'fromUser.id'), 'id')
        test = remove_col(remove_col(test, 'fromUser.id'), 'id')

        x_train, y_train = x_y(train, 'readBy')
        x_dev, y_dev = x_y(dev, 'readBy')
        x_test, y_test = x_y(test, 'readBy')

        logger.info(
            'testing max_depth=6, max_leaf_nodes=31'
        )
        model = DecisionTreeRegressor(max_depth=6,
                                      max_leaf_nodes=31)
        model.fit(x_train, y_train)

        train_predict = model.predict(x_train)
        dev_predict = model.predict(x_dev)
        train_score = mean_squared_error(y_train, train_predict)
        dev_score = mean_squared_error(y_dev, dev_predict)

        print({'mse train': train_score, 'mse test': dev_score})

        test_score = mean_squared_error(y_test, model.predict(x_test))
        print(f'test mse: {test_score}')


def main():
    """ runs suite. these methods were ran then tweaked and ran again """
    graph_analysis(get_train(), get_dev())
    isolated_test(get_train(), get_dev(), get_test())
    focused_test(get_train(), get_dev(), get_test())


if __name__ == "__main__":
    main()
