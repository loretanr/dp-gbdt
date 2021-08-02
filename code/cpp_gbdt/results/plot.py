"""Plot results."""

import pandas as pd
import matplotlib.pyplot as plt


SAMPLES = 4177
filename = 'abalone_size_4177_08.02_18:41.csv'

if __name__ == '__main__':
  data = pd.read_csv(filename, usecols=[
      'dataset', 'nb_samples', 'nb_trees', 'privacy_budget', 'mean', 'std'])

  privacy_budgets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3, 4]
  param_values = data.iloc[0]

  plt.clf()
  plt.grid(True)

  values = data[(data['nb_samples'] == SAMPLES)]
  mean = values['mean']
  std = values['std']
  marker = '-^'
  label = 'DPGBDT (DFS)'

  plt.errorbar(privacy_budgets,
                mean,
                yerr=std,
                fmt=marker,
                capsize=3,
                label=label)

  plt.axis([0, int(max(privacy_budgets)), 0, int(max(
      data[data['nb_samples'] == SAMPLES]['mean']) + 5)])
  plt.legend(loc='upper right')
  plt.title('Dataset={0!s}, Samples={1!s}, Trees={2!s}'.format(
      param_values['dataset'], SAMPLES,
      data[data['nb_samples'] == SAMPLES].iloc[0]['nb_trees']))
  plt.xlabel('Privacy budget')
  plt.ylabel('RMSE')
  plt.savefig(filename.rsplit(".", 1 )[0] + '.png', format='png', dpi=600)
  plt.show()
