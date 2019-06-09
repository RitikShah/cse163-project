# =============================================================================
# Constructs a machine learning model trained on the featured data
# =============================================================================

from .utils import remove_col, x_y
from .split import get_train, get_test

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
import graphviz
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# hyper parameters constants
DEPTH = 6
# SAMPLES_SPLIT = 0.13
# SAMPLES_LEAF = 0.02
# WEIGHT_FRACTION_LEAF = 0.02
LEAF_NODES = 31
IMPURITY_DECREASE = 0.03
# FEATURES = 0


def plot_tree(model, X, y):
    """ Plots the model's tree in graphviz | from Hunter's lectures """
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=X.columns,
                               class_names=y.unique(),
                               filled=True, rounded=True,
                               special_characters=True)
    return graphviz.Source(dot_data)


def calc_close(num, a, b):
    """ Calculates how close the predicts are to actual values in a range """
    close = 0
    for i in range(len(a)):
        if (a[i] >= (b[i] - 2)) & (a[i] <= (b[i] + 2)):
            close += 1

    return close


def ml(train, test, num):
    """ Constructs a model with a given training and testing set """
    logger.info('removing id col')
    train = remove_col(train, 'id')
    train = remove_col(train, 'fromUser.id')
    test = remove_col(test, 'id')
    test = remove_col(test, 'fromUser.id')

    logger.info('splitting data')
    x_train, y_train = x_y(train, 'readBy')
    x_test, y_test = x_y(test, 'readBy')

    logger.info('creating model regressor')
    # hyper parameters were tested in an earlier program
    model = DecisionTreeRegressor(
        max_depth=DEPTH,
        max_leaf_nodes=LEAF_NODES,
        min_impurity_decrease=IMPURITY_DECREASE
    )

    logger.info('fitting model')
    model.fit(x_train, y_train)

    logger.info('making predictions about the test set')
    predicts = model.predict(x_test)

    logger.info('calculating error')
    close = calc_close(num, list(y_test), list(predicts))

    print()
    print(f'MSE: {mean_squared_error(y_test, predicts)}')
    print(
        f'Out of {len(list(predicts))}, {close} predicts are close (+- {num})'
    )

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

    plot_tree(model, x_train.append(x_test), y_train.append(y_test)).view()


def main():
    train = get_train()
    test = get_test()
    ml(train, test)


if __name__ == '__main__':
    main()
