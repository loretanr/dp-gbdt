"""Plot results."""

import os
import pandas as pd
import matplotlib.pyplot as plt


privacy_budgets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3, 4]


def create_plot(filename):
    data = pd.read_csv(filename, usecols=[
        'dataset', 'nb_samples', 'nb_trees', 'privacy_budget', 'mean', 'std'])
    param_values = data.iloc[0]
    SAMPLES = data['nb_samples'][0]
    plt.clf()
    plt.grid(True)

    # plot the dp curve
    values_dp = data[(data['nb_samples'] == SAMPLES) & (data['privacy_budget'] != 0)]
    mean = values_dp['mean']
    std = values_dp['std']
    marker = '-^'
    label = 'DPGBDT (DFS)'
    plt.errorbar(values_dp['privacy_budget'], mean, yerr=std,
                 fmt=marker, capsize=3, label=label)

    # plot the non_dp line
    values_no_dp = data[(data['nb_samples'] == SAMPLES) & (data['privacy_budget'] == 0)]
    if not values_no_dp.empty:
        mean = [values_no_dp.at[0,'mean']] * values_dp.shape[0]
        std = [values_no_dp.at[0,'std']] * values_dp.shape[0]
        marker = '-v'
        label = 'GBDT (DFS) no-dp'
        plt.errorbar(values_dp['privacy_budget'], mean, yerr=std,
                    fmt=marker, capsize=3, label=label)

    plt.axis([0, int(max(privacy_budgets)), 0, int(max(
        data[data['nb_samples'] == SAMPLES]['mean']) * 1.1)])
    plt.legend(loc='upper right')
    plt.title('Dataset={0!s}, Samples={1!s}, Trees={2!s}'.format(
        param_values['dataset'].split('_')[0], SAMPLES,
        data[data['nb_samples'] == SAMPLES].iloc[0]['nb_trees']))
    plt.xlabel('Privacy budget')
    plt.ylabel('RMSE')
    plt.savefig(filename.rsplit(".", 1)[0] + '.png', format='png', dpi=600)
    # plt.show()


if __name__ == '__main__':
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".csv"):
            if not os.path.exists(os.getcwd() + "/" + filename.rsplit(".", 1)[0] + ".png"):
                create_plot(filename)
