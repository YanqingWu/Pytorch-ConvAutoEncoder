import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold.t_sne import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def get_datasets():
    ok = pd.read_csv('tmp/ok.csv', index_col=0)
    ng = pd.read_csv('tmp/ng.csv', index_col=0)
    ok['target'] = 0
    ng['target'] = 1

    data = pd.concat([ok, ng])
    data.reset_index(inplace=True, drop=True)
    # rand = np.random.choice(range(len(data)), 1000)
    # data = data.iloc[rand]

    X = data.drop(columns=['target'])
    y = data['target']
    train_X, val_X, train_y, val_y = train_test_split(X, y)

    return train_X, val_X, train_y, val_y


def train_classifier(train_X, val_X, train_y, val_y):
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(train_X, train_y)
    acc = rf.score(val_X, val_y)
    return acc


def plot_tsne(X, y):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, label="t-SNE")
    plt.show()


if __name__ == '__main__':
    train_X, val_X, train_y, val_y = get_datasets()
    plot_tsne(train_X, train_y)

