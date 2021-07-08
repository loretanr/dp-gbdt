# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Plot problem difficulty (i.e. error) for the baseline on synthetic
datasets.
"""

import pandas as pd
import matplotlib.pyplot as plt

DATASETS = ['synthetic_A', 'synthetic_B', 'synthetic_C', 'synthetic_D']
SAMPLES = [300, 5000, 15000, 25000, 50000, 75000, 100000]


if __name__ == '__main__':
  data = pd.read_csv('./data.csv')
  for dataset in DATASETS:
    plt.clf()
    plt.cla()
    plt.grid(True)
    f, (ax, ax2) = plt.subplots(1, 2, sharey='all', facecolor='w')
    score_loss = data[(data['dataset'] == dataset) &
                      (data['label'] == 'loss')]['mean']
    std_loss = data[(data['dataset'] == dataset) &
                    (data['label'] == 'loss')]['std']

    score_cost = data[(data['dataset'] == dataset) &
                      (data['label'] == 'cost')]['mean']
    std_cost = data[(data['dataset'] == dataset) &
                    (data['label'] == 'cost')]['std']
    ax.errorbar(SAMPLES,
                score_loss,
                yerr=std_loss,
                fmt='-o',
                capsize=3,
                label='Target: loss probability')
    ax.errorbar(SAMPLES,
                score_cost,
                yerr=std_cost,
                fmt='-o',
                capsize=3,
                label='Target: cost (k$)')

    ax2.errorbar(SAMPLES,
                 score_loss,
                 yerr=std_loss,
                 fmt='-o',
                 capsize=3,
                 label='Target: loss probability')
    ax2.errorbar(SAMPLES,
                 score_cost,
                 yerr=std_cost,
                 fmt='-o',
                 capsize=3,
                 label='Target: cost (k$)')

    ax.set_xlim(0, 16000)
    ax2.set_xlim(23000, 110000)
    ax.set_ylim(0, 100)

    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax2.yaxis.set_visible(False)

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    plt.legend(loc='upper right')
    plt.suptitle('Dataset: {0:s} \n Regressor: '
                 'GradientBoostingRegressor'.format(dataset))
    f.text(0.5, 0.02, 'Number of samples', ha='center')
    f.text(0, 0.5, 'Mean Absolute Percentage Error (%)', va='center',
           rotation='vertical')
    f.savefig('{0:s}_complexity.png'.format(dataset), format='png', dpi=600)
