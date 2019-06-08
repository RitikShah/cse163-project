from machine_learning_test import remove_id, x_y, ask_question
from sklearn.tree import DecisionTreeRegressor
from machine_learning import DEPTH, LEAF_NODES
import matplotlib.pyplot as plt
from split import train, test
import seaborn as sns
import pandas as pd


def final_machine_learning():
    # making predictions for every messages in the testing data
    if ask_question('use pickle? [Y or N]: '):
        data = pd.read_pickle('pickle/5m_pkls/analysis.pkl')
    else:
        DEPTH_value = DEPTH
        LEAF_NODES_value = LEAF_NODES
        test_set = remove_id(test())
        train_set = remove_id(train())
        x_train, y_train = x_y(train_set)
        x_test, y_test = x_y(test_set)
        model = DecisionTreeRegressor(
                max_depth=DEPTH_value,
                max_leaf_nodes=LEAF_NODES_value,
                )
        model.fit(x_train, y_train)
        readBy_predict = model.predict(x_test)
        data = test()
        # breakpoint()
        data['readBy_predict'] = readBy_predict
        data.to_pickle('pickle/5m_pkls/analysis.pkl')
    return data


def process_plot_data(data):
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


def plot_top_excalamation_count(data):
    plt.figure(1)
    sns.barplot(x='fromUser.id', y='exclamationCount', hue='readBy_predict',
                data=data)
    plt.xticks(rotation=-15)
    plt.savefig('graphs/top_5_excalamation_count.png')


def plot_bot_exclamation_count(data):
    plt.figure(2)
    sns.barplot(x='fromUser.id', y='exclamationCount', hue='readBy_predict',
                data=data)
    plt.xticks(rotation=-15)
    plt.savefig('graphs/bottom_5_excalamation_count.png')


def plot_top_noun_ratio(data):
    plt.figure(3)
    sns.barplot(x='fromUser.id', y='nounRatio', hue='readBy_predict',
                data=data)
    plt.xticks(rotation=-15)
    plt.savefig('graphs/top_5_noun_ratio.png')


def plot_bot_noun_ratio(data):
    plt.figure(4)
    sns.barplot(x='fromUser.id', y='nounRatio', hue='readBy_predict',
                data=data)
    plt.xticks(rotation=-15)
    plt.savefig('graphs/bottom_5_noun_ratio.png')


def plot_top_urls_count(data):
    plt.figure(5)
    sns.barplot(x='fromUser.id', y='urlsCount', hue='readBy_predict',
                data=data)
    plt.xticks(rotation=-15)
    plt.savefig('graphs/top_5_urls_count.png')


def plot_bot_urls_count(data):
    plt.figure(6)
    sns.barplot(x='fromUser.id', y='urlsCount', hue='readBy_predict',
                data=data)
    plt.xticks(rotation=-15)
    plt.savefig('graphs/bottom_5_urls_countssw.png')


def main():
    data = final_machine_learning()
    rank = process_plot_data(data)
    top_data = get_top_data(rank, data)
    bot_data = get_bot_data(rank, data)
    plot_top_excalamation_count(top_data)
    plot_bot_exclamation_count(bot_data)
    plot_top_noun_ratio(top_data)
    plot_bot_noun_ratio(bot_data)
    plot_top_urls_count(top_data)
    plot_bot_urls_count(bot_data)


if __name__ == '__main__':
    main()
