"""Plot results."""

import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



# privacy_budgets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3, 4]
privacy_budgets = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3,4,5,6,7,8,9,10]


def create_plot(filename):

    def my_x_formatter(x, pos):
        # more ticks than labels
        if x not in [0.1,0.2,0.3,1,2,3,4,5,10]:
            return ""
        val_str = '{:g}'.format(x)
        if val_str.endswith(".0"):
            return val_str.replace(".0", "")
        else:
            return val_str

    def my_y_formatter(x, pos):
        # more ticks than labels
        # if x % 2 != 0:
        #     return ""
        val_str = '{:g}'.format(x)

        return val_str

    data = pd.read_csv(filename, usecols=[
        'dataset', 'nb_samples', 'nb_trees', 'privacy_budget', 'mean', 'std'])
    param_values = data.iloc[0]
    SAMPLES = data['nb_samples'][0]

    fig, ax = plt.subplots(1,1)

    # plt.clf()
    plt.grid(True)

    # plot the baseline
    plt.axhline(y = 3.2205, color = 'mediumseagreen', linestyle = '--')
    plt.annotate('baseline', xy=(0.11, 3.3), xycoords='data', color = 'mediumseagreen')

    # plot the dp curve
    values_dp = data[(data['nb_samples'] == SAMPLES) & (data['privacy_budget'] != 0)]
    mean = values_dp['mean']
    std = values_dp['std']
    marker = '-^'
    label = 'DP-GBDT'
    plt.errorbar(values_dp['privacy_budget'], mean, yerr=std,
                 fmt=marker, capsize=3, label=label, markersize=1, color='royalblue', ecolor='dimgrey')

    # plot the vanilla (non_dp) curve
    values_no_dp = data[(data['nb_samples'] == SAMPLES) & (data['privacy_budget'] == 0)]
    if not values_no_dp.empty:
        mean = [values_no_dp.at[0,'mean']] * values_dp.shape[0]
        std = [values_no_dp.at[0,'std']] * values_dp.shape[0]
        marker = '-v'
        label = 'GBDT no-dp'
        plt.errorbar(values_dp['privacy_budget'], mean, yerr=std,
                    fmt=marker, capsize=3, label=label, markersize=1, color='indianred', ecolor='dimgrey')

    plt.axis([0.09, float(max(privacy_budgets))+1, 1, float(max(
        data[data['nb_samples'] == SAMPLES]['mean']) + max(
        data[data['nb_samples'] == SAMPLES]['std']) + 0.3) ])
    plt.xscale('log', base=2)
    ax.xaxis.set_major_formatter(my_x_formatter)
    ax.yaxis.set_major_formatter(my_y_formatter)
    plt.xticks([elem for elem in privacy_budgets if elem != 0 and elem != 1.5 and elem != 2.5])
    plt.yticks(range(1, math.ceil(float(max( data[data['nb_samples'] == SAMPLES]['mean']) + max (data[data['nb_samples'] == SAMPLES]['std'])))))
    # plt.ticklabel_format(axis='x', style='plain')
    plt.legend(loc='upper right')
    plt.title('dataset={0!s}, samples={1!s}, trees={2!s}'.format(
        param_values['dataset'].split('_')[0], SAMPLES,
        data[data['nb_samples'] == SAMPLES].iloc[0]['nb_trees']))
    plt.xlabel('privacy budget')
    plt.ylabel('RMSE')
    plt.savefig(filename.rsplit(".", 1)[0] + '_xscale.png', format='png', dpi=600)
    # plt.show()


if __name__ == '__main__':
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".csv"):
            if not os.path.exists(os.getcwd() + "/" + filename.rsplit(".", 1)[0] + ".png"):
                create_plot(filename)
