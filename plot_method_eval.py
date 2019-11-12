#! /usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_method_eval_from_csv(csv):
    scores = pd.read_csv(csv)
    print(scores)
    plt.figure(1)
    for i in range(0, scores.shape[1] - 1):
        plt_num = 331 + i
        plt.subplot(plt_num)
        plt.plot(scores.loc[:, 'k'], scores.iloc[:, i+1])
        plt.title('{} graphed over varying eps=k/200'.format(scores.columns[i+1]))
    plt.show()

if __name__ == '__main__':
    #plot_method_eval_from_csv('result_csvs/kmeans_method_eval_scores.csv')
    #plot_method_eval_from_csv('result_csvs/agglom_method_eval_scores.csv')
    plot_method_eval_from_csv('result_csvs/dbscan_method_eval_scores.csv')
