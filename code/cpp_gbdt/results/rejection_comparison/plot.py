"""Plot results."""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

privacy_budgets = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3,4,5,6,7,8,9,10]

def determine_min_max_y(filename):
    data = pd.read_csv(filename, usecols=['mean', 'std'])
    ymin = min(data['mean'])
    ymax = max(data['mean'])
    ymin -= data.loc[data['mean'] == ymin]['std']
    ymax += data.loc[data['mean'] == ymax]['std']
    return float(ymin), float(ymax)

def create_plot(filename):

    def my_x_formatter(x, pos):
        # more ticks than labels
        if x not in [0.1,0.2,0.3,0.5,1,2,3,4,5,6,10]:
            return ""
        val_str = '{:g}'.format(x)
        if val_str.endswith(".0"):
            return val_str.replace(".0", "")
        else:
            return val_str

    def my_y_formatter(x, pos):
        if x > 10:
            if x % 2 != 0:
                return ""
        val_str = '{:g}'.format(x)

        return val_str

    data = pd.read_csv(filename, usecols=[
        'dataset', 'nb_samples', 'nb_trees', 'privacy_budget', 'mean', 'std', 'glc', 'gdf','rejection'])
    param_values = data.iloc[0]
    SAMPLES = data['nb_samples'][0]

    fig, ax = plt.subplots(1,1)

    # plt.clf()
    # plt.grid(True)
    plt.grid(True, color='w', linestyle='-', linewidth=0.4)
    plt.gca().patch.set_facecolor('0.85')

    # plot the baseline
    plt.axhline(y = 3.2205, color = 'mediumseagreen', linestyle = '--')
    plt.annotate('baseline (mean)', xy=(0.1, 2.96), xycoords='data', color = 'mediumseagreen')


    # plot the normal curve
    values_dp = data[(data['nb_samples'] == SAMPLES) & (data['privacy_budget'] != 0) & (data['rejection'] == False) ]
    mean = values_dp['mean']
    std = values_dp['std']
    marker = '-^'
    label = 'DP-GBDT normal/legal'
    plt.errorbar(values_dp['privacy_budget'], mean, yerr=std,
                 fmt=marker, capsize=3, label=label, markersize=1, color='cadetblue', ecolor='dimgrey', elinewidth=1)
    
    # plot the rejection curve
    values_dp = data[(data['nb_samples'] == SAMPLES) & (data['privacy_budget'] != 0) & (data['rejection'] == True) ]
    mean = values_dp['mean']
    std = values_dp['std']
    marker = '-^'
    label = 'DP-GBDT tree-rejection/illegal'
    plt.errorbar(values_dp['privacy_budget'], mean, yerr=std,
                 fmt=marker, capsize=3, label=label, markersize=1, color='darkslateblue', ecolor='dimgrey', elinewidth=1)

    # plot the vanilla (non_dp) curve
    values_no_dp = data[(data['nb_samples'] == SAMPLES) & (data['privacy_budget'] == 0)]
    if not values_no_dp.empty:
        mean = [values_no_dp.at[0,'mean']] * values_dp.shape[0]
        std = [values_no_dp.at[0,'std']] * values_dp.shape[0]
        marker = '-v'
        label = 'GBDT no-dp'
        plt.errorbar(values_dp['privacy_budget'], mean, yerr=std,
                    fmt=marker, capsize=3, label=label, markersize=1, color='indianred', ecolor='dimgrey', elinewidth=1)

    plt.axis([0.09, float(max(privacy_budgets))+1, ymin, ymax])

    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    ax.xaxis.set_major_formatter(my_x_formatter)
    ax.yaxis.set_major_formatter(my_y_formatter)

    plt.xticks([elem for elem in privacy_budgets if elem != 0 and elem != 1.5 and elem != 2.5])
    plt.yticks(np.arange(ymin, ymax, 1))

    plt.legend(loc='upper right')
    plt.title('dataset={0!s}, samples={1!s}, trees={2!s}'.format(
        param_values['dataset'].split('_')[0], SAMPLES,
        data[data['nb_samples'] == SAMPLES].iloc[0]['nb_trees']))
    plt.xlabel('privacy budget')
    plt.ylabel('RMSE')
    plt.savefig(filename.rsplit(".", 1)[0] + '_rmse.pdf', format='pdf', dpi=1200)
    plt.savefig(filename.rsplit(".", 1)[0] + '_rmse.png', format='png', dpi=1200)
    # plt.show()


if __name__ == '__main__':

    ymin = 10000000
    ymax = 0
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".csv"):
            y_min, y_max = determine_min_max_y(filename)
            ymin = min(y_min,ymin)
            ymax = max(y_max,ymax)

    ymin = 2
    ymax = 10

    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".csv"):
            # if not os.path.exists(os.getcwd() + "/" + filename.rsplit(".", 1)[0] + ".png"):
            create_plot(filename)
