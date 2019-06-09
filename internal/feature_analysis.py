# =============================================================================
# Performs feature analysis by created several graphs analyzing the most
#   popular features determined by the machine learning model
# =============================================================================

from .utils import remove_col, x_y, ask_question, unpickle
from .machine_learning import DEPTH, LEAF_NODES
from .split import split

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

FEATURED_PKL = 'pickles/featured.pkl'


def final_machine_learning():
    """ making predictions for every messages in the testing data """
    if ask_question('Use pickle? [Y or N]: '):
        data = pd.read_pickle('pickles/analysis.pkl')
    else:
        train, _, test = split(unpickle(FEATURED_PKL))
        DEPTH_value = DEPTH
        LEAF_NODES_value = LEAF_NODES
        test_set = remove_col(remove_col(test, 'id'), 'fromUser.id')
        train_set = remove_col(remove_col(train, 'id'), 'fromUser.id')
        x_train, y_train = x_y(train_set, 'readBy')
        x_test, y_test = x_y(test_set, 'readBy')
        model = DecisionTreeRegressor(
                max_depth=DEPTH_value,
                max_leaf_nodes=LEAF_NODES_value,
                )
        model.fit(x_train, y_train)
        readBy_predict = model.predict(x_test)
        data = test
        # breakpoint()
        data['readBy_predict'] = readBy_predict
        data.to_pickle('pickles/analysis.pkl')
    return data


def process_plot_data(data):
    """ Processes plot data """
    data['count'] = 1
    count = data.groupby('fromUser.id')['count'].sum()
    users = data.groupby('fromUser.id')['readBy_predict'].mean()
    user_df = pd.DataFrame(dict(count=count, readBy_predict=users)) \
        .reset_index()
    # breakpoint()
    rank_df = user_df.sort_values(['readBy_predict'], ascending=False)
    rank_df_active = rank_df[rank_df['count'] >= 5].reset_index()
    return rank_df_active


def get_top_data(active_rank, data):
    """ Grabs the top 5 users """
    top_5_list = list(active_rank.loc[0:4, 'fromUser.id'])
    top_5_df = data[data['fromUser.id'].isin(top_5_list)].reset_index()
    top_5_df_mean = top_5_df
    top_exclamation_mean = top_5_df.groupby('fromUser.id')['readBy_predict'] \
        .mean()
    for i in range(len(top_5_df_mean)):
        top_5_df_mean.loc[i, 'readBy_predict'] = \
            top_exclamation_mean[top_5_df_mean.loc[i, 'fromUser.id']]
    return top_5_df_mean


def get_bot_data(active_rank, data):
    """ Grabs the bottom 5 users """
    bottom_5_list = list(active_rank.loc[len(active_rank) - 5:
                                         len(active_rank), 'fromUser.id'])
    bottom_5_df = data[data['fromUser.id'].isin(bottom_5_list)].reset_index()
    bot_exclamation_mean = bottom_5_df \
        .groupby('fromUser.id')['readBy_predict'].mean()
    bottom_5_df_mean = bottom_5_df
    for i in range(len(bottom_5_df_mean)):
        bottom_5_df_mean.loc[i, 'readBy_predict'] = \
            bot_exclamation_mean[bottom_5_df_mean.loc[i, 'fromUser.id']]
    return bottom_5_df_mean


def plot_top_mention_count(data):
    """ Plots various ratios/scores about top and bottom users """
    plt.figure(1, figsize=(20, 20))
    sns.catplot(x='fromUser.id', y='mentionsCount', hue='readBy_predict',
                data=data, kind="bar", legend_out=True)
    plt.ylim(0, 12.0)
    plt.xticks(rotation=-10)
    plt.title("top user's mention count graph")
    plt.savefig('graphs/top_5_mention_count.png')


def plot_bot_mention_count(data):
    """ Plots various ratios/scores about top and bottom users """
    plt.figure(2, figsize=(20, 20))
    sns.catplot(x='fromUser.id', y='mentionsCount', hue='readBy_predict',
                data=data, kind="bar", legend_out=True)
    plt.ylim(0, 12.0)
    plt.xticks(rotation=-10)
    plt.title("bottom user's mention count graph")
    plt.savefig('graphs/bottom_5_mention_count.png')


def plot_top_excalamation_count(data):
    """ Plots various ratios/scores about top and bottom users """
    plt.figure(1)
    sns.catplot(x='fromUser.id', y='exclamationCount', hue='readBy_predict',
                data=data, kind="bar", legend_out=True)
    plt.ylim(0, 6.0)
    plt.xticks(rotation=-10)
    plt.title("top user's excalamation count graph")
    plt.savefig('graphs/top_5_excalamation_count.png')


def plot_bot_exclamation_count(data):
    """ Plots various ratios/scores about top and bottom users """
    plt.figure(2)
    sns.catplot(x='fromUser.id', y='exclamationCount', hue='readBy_predict',
                data=data, kind="bar", legend_out=True)
    plt.ylim(0, 6.0)
    plt.xticks(rotation=-10)
    plt.title("bottom user's excalamation count graph")
    plt.savefig('graphs/bottom_5_excalamation_count.png')


def plot_top_polarity(data):
    """ Plots various ratios/scores about top and bottom users """
    plt.figure(5)
    sns.catplot(x='fromUser.id', y='polarity', hue='readBy_predict',
                data=data, kind="bar", legend_out=True)
    plt.xticks(rotation=-10)
    plt.ylim(-1.0, 1.0)
    plt.title("top user's polarity graph")
    plt.savefig('graphs/top_5_polarity.png')


def plot_bot_polarity(data):
    """ Plots various ratios about top and bottom users """
    plt.figure(6)
    sns.catplot(x='fromUser.id', y='polarity', hue='readBy_predict',
                data=data, kind="bar", legend_out=True)
    plt.xticks(rotation=-10)
    plt.ylim(-1.0, 1.0)
    plt.title("bottom user's polarity graph")
    plt.savefig('graphs/bottom_5_polarity.png')


def plot_top_word_count(data):
    """ Plots various ratios/scores about top and bottom users """
    plt.figure(5)
    sns.catplot(x='fromUser.id', y='wordCount', hue='readBy_predict',
                data=data, kind="bar", legend_out=True)
    plt.xticks(rotation=-10)
    plt.ylim(0, 3000.0)
    plt.title("top user's word count graph")
    plt.savefig('graphs/top_5_word_count.png')


def plot_bot_word_count(data):
    """ Plots various ratios/scores about top and bottom users """
    plt.figure(6)
    sns.catplot(x='fromUser.id', y='wordCount', hue='readBy_predict',
                data=data, kind="bar", legend_out=True)
    plt.xticks(rotation=-10)
    plt.ylim(0, 3000.0)
    plt.title("bottom user's word count graph")
    plt.savefig('graphs/bottom_5_word_count.png')


def plot_top_noun_ratio(data):
    """ Plots various ratios/scores about top and bottom users """
    plt.figure(3)
    sns.catplot(x='fromUser.id', y='nounRatio', hue='readBy_predict',
                data=data, kind="bar", legend_out=True)
    plt.ylim(0, 0.3)
    plt.xticks(rotation=-10)
    plt.title("top user's noun ratio graph")
    plt.savefig('graphs/top_5_noun_ratio.png')


def plot_bot_noun_ratio(data):
    """ Plots various ratios/scores about top and bottom users """
    plt.figure(4)
    sns.catplot(x='fromUser.id', y='nounRatio', hue='readBy_predict',
                data=data, kind="bar", legend_out=True)
    plt.ylim(0, 0.3)
    plt.xticks(rotation=-10)
    plt.title("bottom user's noun ratio graph")
    plt.savefig('graphs/bottom_5_noun_ratio.png')


def plot_top_subjectivity(data):
    """ Plots various ratios/scores about top and bottom users """
    plt.figure(5)
    sns.catplot(x='fromUser.id', y='subjectivity', hue='readBy_predict',
                data=data, kind="bar", legend_out=True)
    plt.xticks(rotation=-10)
    plt.ylim(-1.0, 1.0)
    plt.title("top user's subjectivity graph")
    plt.savefig('graphs/top_5_subjectivity.png')


def plot_bot_subjectivity(data):
    """ Plots various ratios/scores about top and bottom users """
    plt.figure(6)
    sns.catplot(x='fromUser.id', y='subjectivity', hue='readBy_predict',
                data=data, kind="bar", legend_out=True)
    plt.xticks(rotation=-10)
    plt.ylim(-1.0, 1.0)
    plt.title("bottom user's subjectivity graph")
    plt.savefig('graphs/bottom_5_subjectivity.png')


def main():
    """ Runs entire plotting suit """
    data = final_machine_learning()
    rank = process_plot_data(data)
    top_data = get_top_data(rank, data)
    bot_data = get_bot_data(rank, data)
    sns.set(rc={'figure.figsize': (12, 12)})
    # breakpoint()
    if ask_question('plot mention count? [Y or N]: '):
        plot_top_mention_count(top_data)
        plot_bot_mention_count(bot_data)
    if ask_question('plot excalmation count? [Y or N]: '):
        plot_top_excalamation_count(top_data)
        plot_bot_exclamation_count(bot_data)
    if ask_question('plot  polarity? [Y or N]: '):
        plot_top_polarity(top_data)
        plot_bot_polarity(bot_data)
    if ask_question('plot word count? [Y or N]: '):
        plot_top_word_count(top_data)
        plot_bot_word_count(bot_data)
    if ask_question('plot noun ratio? [Y or N]: '):
        plot_top_noun_ratio(top_data)
        plot_bot_noun_ratio(bot_data)
    if ask_question('plot subjectivity? [Y or N]: '):
        plot_top_subjectivity(top_data)
        plot_bot_subjectivity(bot_data)


if __name__ == '__main__':
    main()
