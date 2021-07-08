# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Plot results."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from evaluation import estimator

INPUT_NAME = 'results_alltrees_40trees_23-06-21_16:48.csv'
PATH = './results/abalone/'
SAMPLES = [4177]
PRIVACY_BUDGETS = [1,2,4,6,8,10] # np.arange(0.1, 1.0, 0.1) #[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]#[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5] # = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.5, 3, 3.5, 4] # np.arange(0.1, 1.0, 0.1)

if __name__ == '__main__':
  data = pd.read_csv(PATH + INPUT_NAME, usecols=[
      'dataset', 'nb_samples', 'privacy_budget', 'nb_tree',
      'nb_tree_per_ensemble', 'max_depth',
      'max_leaves', 'learning_rate', 'nb_of_runs', 'mean', 'std', 'model',
      'config', 'balance_partition'])

  param_values = data.iloc[0]

  # Own model
  models = [estimator.DPGBDT]

  for nb_samples in SAMPLES:
    plt.clf()
    plt.grid(True)
    for model in models:
      model_name = str(model).split('.')[-1][:-2]
      for config in [ 'DFS', '3-trees', 'Vanilla']:
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
            len(PRIVACY_BUDGETS))])
          std = list([values['std'].values[0] for _ in range(
            len(PRIVACY_BUDGETS))])
          label = 'GBDT (Vanilla)'
          marker = '-ko'

        plt.errorbar(PRIVACY_BUDGETS,mean,yerr=std,fmt=marker,capsize=3,label=label)
    xaxis_upperlimit = PRIVACY_BUDGETS[-1] + round(0.1 * (PRIVACY_BUDGETS[-1] - PRIVACY_BUDGETS[0]),2)
    xaxis_lowerlimit = PRIVACY_BUDGETS[0] - round(0.1 * (PRIVACY_BUDGETS[-1] - PRIVACY_BUDGETS[0]),2)

    plt.axis([xaxis_lowerlimit, xaxis_upperlimit,
        0, int(max(data[data['nb_samples'] == nb_samples]['mean']) + max(
            data[data['nb_samples'] == nb_samples]['std']) + 5)])
    plt.legend(loc='upper right')
    plt.title('Dataset={0!s}, Samples={1!s}, Trees={2!s}, alltrees'.format(
        param_values['dataset'], nb_samples,
        data[data['nb_samples'] == nb_samples].iloc[0]['nb_tree']))
    plt.xlabel('Privacy budget')
    plt.ylabel('RMSE')
    now = datetime.now().strftime("%d-%m-%y_%H:%M")
    plt.savefig(
        PATH + 'results_alltrees_40trees_{0!s}_{1!s}.png'.format(nb_samples, now), 
        format='png', dpi=600)