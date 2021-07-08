# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Plot histograms for synthetic data."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
  for l in ['A', 'B', 'C', 'D']:
    data = pd.read_csv('../src/synthetic/synthetic_{0:s}.csv'.format(l))
    y_loss = np.asarray(data['company_suffers_a_loss'].values)
    y_cost = np.asarray(data['cost_of_loss'].values)

    f, (ax0, ax1) = plt.subplots(1, 2)

    ax0.hist(y_loss, bins='auto')
    ax0.set_xlim([0, 1])
    ax0.set_ylabel('Count')
    ax0.set_xlabel('Target')
    ax0.set_title('Loss (probability)')

    ax1.hist(y_cost, bins='auto')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Target')
    ax1.set_title('Cost (k$)')

    f.suptitle('Dataset: synthetic_{0:s}'.format(l), y=0.035)
    f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    f.savefig('./synthetic_{0:s}_histogram.png'.format(l),
              format='png',
              dpi=600)
