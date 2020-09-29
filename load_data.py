from settings import *
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# loads data from csv file and prints info about the dataset
def load_data(csv_path=TRAINING_SET):
    train_data = pd.read_csv(csv_path)
    return train_data


# splits the dataframe into a training set and a test set.
def split_dataset(data_frame):
    train_set, test_set = train_test_split(data_frame, test_size=0.2, shuffle=True, random_state=23)
    return {
        'train_set': train_set,
        'test_set': test_set
    }


def plot_heatmap(data_frame):
    corr = data_frame.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    crit_temp_series = corr["critical_temp"].sort_values(ascending=False)
    print(type(crit_temp_series))
    print(corr["critical_temp"].sort_values(ascending=False))
    print(type(corr))
    # add the other plot here
    fig = plt.subplots(figsize=(10, 10))
    sn.heatmap(corr, mask=mask, annot=False, linewidths=0.2, square=True)
    plt.show()


def plot_temp_corr(data_frame):
    corr = data_frame.corr()
    crit_temp_series = corr["critical_temp"].sort_values(ascending=False)
    print(type(crit_temp_series))
    print(crit_temp_series)
    # add the other plot here
    crit_temp_series.plot(kind="barh", figsize=(20, 10), fontsize=8)
    plt.show()


def plot_dist(data_frame):
    critical_temps = data_frame['critical_temp']
    print('Median critical temperature', critical_temps.median(), 'Mean critical temperature', critical_temps.mean())
    sn.distplot(critical_temps)
    plt.show()


def load_and_split(plot_graphs=False):
    df = load_data()
    data_dict = split_dataset(df)
    if plot_graphs:
        train_set = data_dict.get('train_set')
        df_train = pd.DataFrame(train_set)
        plot_heatmap(df_train)
        plot_temp_corr(df_train)
        plot_dist(df_train)
    return data_dict
