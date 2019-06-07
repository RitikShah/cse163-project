from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from split import train, dev, test
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def ask_question(s):
    return str(input(s)).upper()[0] == 'Y'


def remove_id(dset):
    return dset.loc[:, dset.columns != 'id']


def x_y(dset):
    return dset.loc[:, dset.columns != 'readBy'], dset['readBy']


def depth(x_train, y_train, x_dev, y_dev, x_test, y_test):
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


def impurity_split(x_train, y_train, x_dev, y_dev, x_test, y_test):
    min_mse = None
    min_model = None
    min_depth = 0

    for i in range(1, 100, 10):
        model = DecisionTreeRegressor(min_impurity_split=i)
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
    print(f'min_impurity_split: {min_depth}')

    test_score = mean_squared_error(y_test, min_model.predict(x_test))
    print(f'test mse: {test_score}')


def isolated_test(train, dev, test):
    logger.info('focused test 4')
    train = remove_id(train)
    dev = remove_id(dev)
    test = remove_id(test)

    x_train, y_train = x_y(train)
    x_dev, y_dev = x_y(dev)
    x_test, y_test = x_y(test)

    depth(x_train, y_train, x_dev, y_dev, x_test, y_test)
    # leaf_nodes(x_train, y_train, x_dev, y_dev, x_test, y_test)
    # impurity_decrease(x_train, y_train, x_dev, y_dev, x_test, y_test)


def focused_test(train, dev, test):
    logger.info('focused test 5')
    train = remove_id(train)
    dev = remove_id(dev)
    test = remove_id(test)

    x_train, y_train = x_y(train)
    x_dev, y_dev = x_y(dev)
    x_test, y_test = x_y(test)

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
    # isolated_test(train(), dev(), test())
    focused_test(train(), dev(), test())


if __name__ == "__main__":
    main()
