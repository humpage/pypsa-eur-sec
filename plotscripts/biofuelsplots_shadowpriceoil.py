import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

scenario = '37h_full_fixedIndustry_2060'
sdir = '../results/{}/csvs/costs.csv'.format(scenario)
pricedir = '../results/{}/csvs/prices.csv'.format(scenario)
output = '../results/37h_full_fixedIndustry_2060_shadowprice'
balances = '../results/{}/csvs/supply_energy.csv'.format(scenario)




def axes_handling_left(ax):
    handles, labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    # ax.set_ylim([0,snakemake.config['plotting']['costs_max']])

    ax.set_ylabel("System Cost [EUR billion per year]")

    ax.set_xlabel("")

    ax.grid(axis='x')

    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)
    ax.get_legend().remove()


def axes_handling_right(ax, legend=False):
    handles, labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    # ax.set_ylim([0,snakemake.config['plotting']['costs_max']])

    ax.set_xlabel("")

    ax.grid(axis='x')

    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)

    if legend == False:
        ax.get_legend().remove()


def place_subplot(df, ax, ylabel, transportOnly=True, legend=False):

    print(df.T)
    df.T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        # color=colors,  # [snakemake.config['plotting']['tech_colors'][i] for i in new_index]
    )

    handles, labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_ylabel(ylabel)  # "System Cost [EUR billion per year]")
    ax.set_xlabel("Biofuel mandate")

    ax.grid(axis='x')
    ax.set_ylim([0, 55])  # snakemake.config['plotting']['costs_max']])

    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)
    if legend == False:
        ax.get_legend().remove()


def rename_index(df):
    for num in np.arange(0, 9):
        if num == 0:
            df.index = df.index.str.replace('.*B0p{}.*'.format(str(num)), 'Opt')
        else:
            df.index = df.index.str.replace('.*B0p{}.*'.format(str(num)), '>{}0%'.format(num))
        df.index = df.index.str.replace('.*B1p0.*', '100%')


def plot_scenarios(output, n_range, transportOnly):
    price_df = pd.read_csv(pricedir,  # skiprows=2,
                          index_col=list(range(1)),
                          header=list(range(n_range))
                          )
    print(price_df)
    print(price_df.index)
    print(price_df.loc['oil'])

    df = price_df.loc['oil']#.groupby(price_df.loc['oil'].index.get_level_values(3))#.sum()
    print(df)
    print(df.index)
    df.index = df.index.map('|'.join).str.strip('|')
    print(df)
    # convert to billions
    # df = df / 1e9


    # df = df.drop(to_drop)

    # print(df.sum())
    df2 = df.filter(regex='High.*S400')#.*B0.*Ef2.*E2.*C2.*CS2.*O2.*I0')
    df3 = df.filter(regex='High.*S1500')#.*B0.*Ef2.*E2.*C2.*CS2.*O2.*I0')
    df4 = df.filter(regex='Med.*S400')#.*B0.*Ef2.*E2.*C2.*CS2.*O2.*I0')
    df5 = df.filter(regex='Med.*S1500')#.*B0.*Ef2.*E2.*C2.*CS2.*O2.*I0')

    rename_index(df2)
    rename_index(df3)
    rename_index(df4)
    rename_index(df5)

    # print(df.sum())
    fig, ((ax2, ax3), (ax4, ax5)) = plt.subplots(2, 2, figsize=(12, 8))

    place_subplot(df2, ax2, 'Hög biomassa, låg CO2-lagring', transportOnly)
    place_subplot(df3, ax3, 'Hög biomassa, hög CO2-lagring', transportOnly, legend=True)
    place_subplot(df4, ax4, 'Låg biomassa, låg CO2-lagring', transportOnly)
    place_subplot(df5, ax5, 'Hög biomassa, hög CO2-lagring', transportOnly)

    # axes_handling_left(ax2)
    # axes_handling_left(ax4)
    # axes_handling_right(ax3,legend=True)
    # axes_handling_right(ax5)

    if transportOnly:
        output = output + '_transportOnly'

    fig.savefig(output + '.pdf', bbox_inches='tight')


n_range = 4
# Get shares of resources to fuel production
# plot_balances(balances)
plot_scenarios(output, n_range, transportOnly=False)
# plot_scenarios(output, transportOnly=True)
