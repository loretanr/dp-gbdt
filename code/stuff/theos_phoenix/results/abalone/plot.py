# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Plot results."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluation import estimator

PATH = './results/abalone/'
NAME = 'results_PHOENIX_04-08-21_16:19.csv'
SAMPLES = [4177]

if __name__ == '__main__':
  data = pd.read_csv(PATH + NAME, usecols=[
      'dataset', 'nb_samples', 'privacy_budget', 'nb_tree',
      'nb_tree_per_ensemble', 'max_depth',
      'max_leaves', 'learning_rate', 'nb_of_runs', 'mean', 'std', 'model',
      'config', 'balance_partition'])

  # privacy_budgets = [0.1, 0.3, 0.6, 1, 1.5, 2, 3, 5, 7, 9]
  privacy_budgets = [0.1, 0.5, 1, 3, 8]
  param_values = data.iloc[0]

  # Own model
  models = [estimator.DPGBDT, estimator.DPRef]

  for nb_samples in SAMPLES:
    plt.clf()
    plt.grid(True)
    for model in models:
      model_name = str(model).split('.')[-1][:-2]
      for config in ['DFS','Vanilla']:
        if config in ['BFS']:
          continue
        if model_name == 'DPRef' and config != 'DFS':
          continue
        values = data[(data['model'] == model_name) & (
            data['config'] == config) & (data['nb_samples'] == nb_samples)]
        if config != 'Vanilla':
          mean = values['mean']
          std = values['std']
          if model_name == 'DPRef':
            marker = '-v'
            label = 'DPRef'
          else:
            marker = '-^'
            label = 'DPGBDT (DFS)'
        else:
          mean = list([values['mean'].values[0] for _ in range(
            len(privacy_budgets))])
          std = list([values['std'].values[0] for _ in range(
            len(privacy_budgets))])
          label = 'GBDT (Vanilla)'
          marker = '-ko'

        plt.errorbar(privacy_budgets,
                     mean,
                     yerr=std,
                     fmt=marker,
                     capsize=3,
                     label=label)

    plt.axis([0, int(max(privacy_budgets)), 0, int(max(
        data[data['nb_samples'] == nb_samples]['mean']) + max(
            data[data['nb_samples'] == nb_samples]['std']) ) * 1.1 ])
    plt.legend(loc='upper right')
    plt.title('Dataset={0!s}, Samples={1!s}, Trees={2!s}'.format(
        param_values['dataset'], nb_samples,
        data[data['nb_samples'] == nb_samples].iloc[0]['nb_tree']))
    plt.xlabel('Privacy budget')
    plt.ylabel('RMSE')
    plt.savefig(
        PATH + NAME.rsplit('.',1)[0]  + '.png', format='png', dpi=600)
