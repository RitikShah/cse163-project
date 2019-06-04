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
plot_data = []
for i in range(1, 100):
    model = DecisionTreeRegressor(max_depth=i, random_state=1)
    model.fit(x_train, y_train)

    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)

    plot_data.append({'max depth': i, 'train accuracy': train_score,
                      'test accuracy': test_score})

plot_data = pd.DataFrame(plot_data)
fig, (ax1, ax2) = plt.subplots(2, figsize=(30, 30))
sns.relplot(ax=ax1, kind='line', x='max depth', y='train accuracy',
            data=plot_data)
sns.relplot(ax=ax2, kind='line', x='max depth', y='test accuracy',
            data=plot_data)
plt.show()
