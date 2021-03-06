# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Plot results."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from evaluation import estimator

PATH = './results/adult/'
SAMPLES = [5000]

if __name__ == '__main__':
  data = pd.read_csv(PATH + 'data_21-07-21_09:37.csv', usecols=[
      'dataset', 'nb_samples', 'privacy_budget', 'nb_tree',
      'nb_tree_per_ensemble', 'max_depth',
      'max_leaves', 'learning_rate', 'nb_of_runs', 'mean', 'std', 'model',
      'config', 'balance_partition'])

  privacy_budgets = [0.5,2,4,6,8,10]
  param_values = data.iloc[0]

  # Own model
  models = [estimator.DPGBDT]

  for nb_samples in SAMPLES:
    plt.clf()
    plt.grid(True)
    for model in models:
      model_name = str(model).split('.')[-1][:-2]
      for config in ['DFS']:
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
            label = 'DPGBDT_Ref'
          else:
            if config == '3-trees':
              marker = '-s'
              label = 'DPGBDT (2-nodes)'
            else:
              marker = '-^'
              label = 'DPGBDT (Depth-first)'
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

    plt.axis([0, 11, 0, int(max(
        data[data['nb_samples'] == nb_samples]['mean']) + max(
            data[data['nb_samples'] == nb_samples]['std']) + 5)])
    plt.legend(loc='upper right')
    plt.title('Dataset={0!s}, Samples={1!s}, Trees={2!s}'.format(
        param_values['dataset'], nb_samples,
        data[data['nb_samples'] == nb_samples].iloc[0]['nb_tree']))
    plt.xlabel('Privacy budget')
    plt.ylabel('Test Error (%)')
    now = datetime.now().strftime("%d-%m-%y_%H:%M")
    plt.savefig(
        PATH + 'results_21-07-21_09:37{0!s}_{1!s}.png'.format(nb_samples, now),
        format='png', dpi=600)
