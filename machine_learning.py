from machine_learning_test import remove_id, x_y
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
from split import train, test
import graphviz
import logging
import pickle

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# TODO: throw this in a function pls

# hyper parameters constants
DEPTH = 6
# SAMPLES_SPLIT = 0.13
# SAMPLES_LEAF = 0.02
# WEIGHT_FRACTION_LEAF = 0.02
LEAF_NODES = 31
# IMPURITY_DECREASE = 0.03
# FEATURES = 0


def plot_tree(model, X, y):
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=X.columns,
                               class_names=y.unique(),
                               filled=True, rounded=True,
                               special_characters=True)
    return graphviz.Source(dot_data)


def main():
    logger.info('unpickling')
    train_set = remove_id(train())
    test_set = remove_id(test())

    logger.info('splitting data')
    x_train, y_train = x_y(train_set)
    x_test, y_test = x_y(test_set)

    logger.info('creating regressor')
    # hyper parameters were tested in an earlier program
    model = DecisionTreeRegressor(
        max_depth=DEPTH,
        max_leaf_nodes=LEAF_NODES,
    )

    logger.info('fitting model')
    model.fit(x_train, y_train)

    logger.info('making predictions about test set')
    a = list(model.predict(x_test))
    b = list(y_test)

    logger.info('calculating error')
    close = 0
    for i in range(len(a)):
        if (a[i] >= (b[i] - 2)) & (a[i] <= (b[i] + 2)):
            close += 1

    print(f'MSE: {mean_squared_error(y_test, model.predict(x_test))}')
    print('Out of', str(len(b)), ',', str(close), 'predicts are close')

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

    plot_tree(model, x_train.append(x_test), y_train.append(y_test)).view()


if __name__ == '__main__':
    main()
