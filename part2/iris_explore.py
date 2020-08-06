import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets


def bunch_to_df(bunch):
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df['target'] = pd.Series(bunch.target)
    return df

def visualize_df(df):
    pd.plotting.scatter_matrix(df, c=pd.Categorical(df['target']), marker='o')
    plt.show()


iris = sklearn.datasets.load_iris()
iris_df = bunch_to_df(iris)

print(iris_df.head())
print(iris_df.describe())

visualize_df(iris_df)
