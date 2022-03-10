import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

year = 2040
scenario = 'serverResults/mainScenarios{}'.format(year)
sdir = '../results/{}/csvs/costs.csv'.format(scenario)
output = '../results/fuelSupply{}'.format(year)
balances = '../results/{}/csvs/supply_energy.csv'.format(scenario)
metrics = '../results/{}/csvs/metrics.csv'.format(scenario)
costs = '../results/{}/csvs/costs.csv'.format(scenario)
prices_dir = '../results/{}/csvs/prices.csv'.format(scenario)
H2shareofACrevenueCSV = '../results/{}/csvs/H2_share_of_AC_revenue.csv'.format(scenario)


def get_H2shareofACrevenue(H2shareofACrevenueCSV):

    global H2shareofACrevenue

    H2shareofACrevenue_df = pd.read_csv(
        H2shareofACrevenueCSV,
        index_col=list(range(1)),
        header=list(range(n_range))
    )

    H2shareofACrevenue_df.columns = H2shareofACrevenue_df.columns.map('|'.join).str.strip('|')
    H2shareofACrevenue = H2shareofACrevenue_df.loc['H2_share_of_AC_revenue']

    return H2shareofACrevenue


def plot_metrics(metrics):

    global co2_shadowPrice

    metrics_df = pd.read_csv(
        metrics,
        index_col=list(range(1)),
        header=list(range(n_range))
    )

    metrics_df.columns = metrics_df.columns.map('|'.join).str.strip('|')

    co2_shadowPrice = metrics_df.loc['co2_shadow']

    return co2_shadowPrice


def plot_costs(costs):

    global solid_biomass_cost

    costs_df = pd.read_csv(
        costs,
        index_col=list(range(3)),
        header=list(range(n_range))
    )

    costs_df.columns = costs_df.columns.map('|'.join).str.strip('|')

    v = [['generators']]
    for k in v:
        df = costs_df.loc[k]
        df = df.groupby(df.index.get_level_values(2)).sum()

        solid_biomass_cost = df[['solid biomass' in s for s in df.index]].sum() / 1e9

    return solid_biomass_cost


def plot_balances(balances):
    global H2toEfuelShare, ACtoElectrolysisShare, SolidBiomasstoBtLshare, FossilLiquidFuelsShare, CO2toEfuelShare, \
        transportCO2emissionShare, TotalCO2captured, CCUS_DACshare, BiofuelCO2captureShare, FossilLiquidFuelsAmount, \
        BiofuelAmount, ElectrofuelAmount, H2productionTot, SolidBiomassTot

    co2_carriers = ["co2", "co2 stored", "process emissions"]

    balances_df = pd.read_csv(
        balances,
        index_col=list(range(3)),
        header=list(range(n_range))
    )

    balances_df.columns = balances_df.columns.map('|'.join).str.strip('|')
    balances = {i.replace(" ", "_"): [i] for i in balances_df.index.levels[0]}
    balances["energy"] = [i for i in balances_df.index.levels[0] if i not in co2_carriers]

    v = [['AC'], ['H2'], ['solid biomass'], ['oil'], ['co2 stored'], ['co2']]
    for k in v:
        df = balances_df.loc[k]
        df = df.groupby(df.index.get_level_values(2)).sum()

        # convert MWh to TWh
        df = df / 1e6

        # remove trailing link ports
        df.index = [i[:-1] if ((i != "co2") and (i[-1:] in ["0", "1", "2", "3", "4"])) else i for i in df.index]

        df = df.groupby(df.index.map(rename_techs_balances)).sum()

        # print('Dataframe: ', k, df[df < 0].dropna())#.sum())
        if k == ['H2']:
            H2toEfuelShare = df.loc['electrofuel'] / df[df < 0].dropna().sum()
            # print('Efuel: ', H2toEfuelShare)
            H2productionTot = abs(df[df > 0].dropna().sum())

        elif k == ['AC']:
            ACtoElectrolysisShare = df.loc['H2 Electrolysis'] / df[df < 0].dropna().sum()
            # print('Electrolysis: ', ACtoElectrolysisShare)

        elif k == ['solid biomass']:
            SolidBiomasstoBtLshare = abs(df.loc['biomass to liquid'] / df[df > 0].dropna().sum())
            SolidBiomassTot = abs(df[df > 0].dropna().sum())
            # print('BtL: ', SolidBiomasstoBtLshare)

        elif k == ['oil']:
            FossilLiquidFuelsAmount = abs(df.loc['oil'])
            # print('Fossil fuels: ', FossilLiquidFuelsAmount, ' TWh')
            ElectrofuelAmount = abs(df.loc['electrofuel'])
            # print('Efuels: ', ElectrofuelAmount, ' TWh')
            BiofuelAmount = abs(df.loc['biomass to liquid'])
            # print('Biofuels: ', BiofuelAmount, ' TWh')
            FossilLiquidFuelsShare = abs(df.loc['oil'] / df[df < 0].dropna().sum())
            # print('Fossil fuels share: ', FossilLiquidFuelsShare)

        elif k == ['co2 stored']:
            CO2toEfuelShare = df.loc['electrofuel'] / df[df < 0].dropna().sum()
            # print('CO2 to Efuel share: ', CO2toEfuelShare)

            BiofuelCO2captureShare = df.loc['biomass to liquid'] / df[df > 0].dropna().sum()
            # print('Biofuel captured CO2 share: ', BiofuelCO2captureShare)

            CCUS_DACshare = df.loc['DAC'] / df[df > 0].dropna().sum()
            # print('CCUS DAC share: ', CCUS_DACshare)

            TotalCO2captured = df[df > 0].sum()
            # print('Total CO2 captured: ', TotalCO2captured, ' MtCO2')

        elif k == ['co2']:
            transportCO2emissionShare = (df.loc['electrofuel'] + df.loc['biomass to liquid'] + df.loc['oil emissions'] +
                                         df.loc['shipping oil emissions']) / df[df > 0].dropna().sum()
            # print('Transport share of total CO2 emissions: ', transportCO2emissionShare)

    return H2toEfuelShare, ACtoElectrolysisShare, SolidBiomasstoBtLshare, FossilLiquidFuelsShare, CO2toEfuelShare,\
           CCUS_DACshare, transportCO2emissionShare, TotalCO2captured, BiofuelCO2captureShare,\
           FossilLiquidFuelsAmount, BiofuelAmount, ElectrofuelAmount, H2productionTot, SolidBiomassTot


# consolidate and rename
def rename_techs(label):
    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral "
    ]

    rename_if_contains = [
        "CHP",
        "gas boiler",
        "biogas",
        "solar thermal",
        "air heat pump",
        "ground heat pump",
        "resistive heater",
        "Fischer-Tropsch"
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        "battery": "battery storage",
        "CC": "CC"
    }

    rename = {
        "solar": "solar PV",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        "offwind-ac": "offshore wind (AC)",
        "offwind-dc": "offshore wind (DC)",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "co2 Store": "DAC",
        "co2 stored": "CO2 sequestration",
        "AC": "transmission lines",
        "DC": "transmission lines",
        "B2B": "transmission lines"
    }

    simplify = {
        "transmission lines": 'power',
        "hydroelectricity": 'power',
        "hydro reservoir": 'power',
        "run of river": 'power',
        "pumped hydro storage": 'power',
        "onshore wind": 'power',
        "offshore wind": 'power',
        "offshore wind (AC)": 'power',
        "offshore wind (DC)": 'power',
        "solar PV": 'power',
        "solar thermal": 'heat',
        "building retrofitting": 'heat',
        "ground heat pump": 'heat',
        "air heat pump": 'heat',
        "heat pump": 'heat',
        "resistive heater": 'heat',
        "power-to-heat": 'heat',
        "gas-to-power/heat": 'heat',
        "CHP": 'CHP',
        "OCGT": 'CHP',
        "gas boiler": 'heat',
        "hydrogen storage": 'hydrogen derivatives',
        "power-to-gas": 'hydrogen derivatives',
        "H2": 'hydrogen derivatives',
        "H2 liquefaction": 'hydrogen derivatives',
        "landscape care solid biomass": 'solid biomass',
        "forest residues solid biomass": 'solid biomass',
        "industry wood residues solid biomass": 'solid biomass',
        "solid biomass import": 'solid biomass',
        "straw digestible biomass": 'digestible biomass',
        "municipal biowaste digestible biomass": 'digestible biomass',
        "manureslurry digestible biomass": 'digestible biomass',
        "biogas": 'other biomass usage',
        "solid biomass to electricity": 'other biomass usage',
        "BioSNG": 'other biomass usage',
        "digestible biomass to hydrogen": 'other biomass usage',
        'solid biomass to hydrogen': 'other biomass usage',
        "lowT process steam solid biomass": 'other biomass usage',
        "lowT process steam methane": 'industry',
        "lowT process steam electricity": 'industry',
        "battery storage": 'power',
        "hot water storage": 'heat',
        'coal': 'coal',
        'oil': 'fossil liquid fuel',
        'gas': 'fossil gas',
        'lignite': 'coal',
        'uranium': 'power',
        'process emissions': 'industry',
        'gas for industry': 'industry',
        'gas for mediumT industry': 'industry',
        'lowT process steam H2': 'industry',
        'SMR': 'other',
        'CC': 'carbon capture',
        'methanation': 'other',
        'nuclear': 'power',
        'nuclear_new': 'power',
        'lowT process steam heat pump': 'industry',
        'solid biomass for mediumT industry': 'other biomass usage',
        'gas for highT industry': 'industry',
        'BEV charger': 'other',
        'oil boiler': 'heat',
        'sewage sludge digestible biomass': 'digestible biomass'
    }

    simplify_more = {
        'heat': 'other',
        'industry': 'other',
        'CHP': 'other',
        'power': 'power excl. fossil fuels',
        'hydrogen derivatives': 'other',
        'coal': 'fossil fuels',
        # 'digestible biomass': 'biomass',
        # 'solid biomass': 'biomass',
        'biomass to liquid': 'biofuel',
        'DAC': 'DAC',
        'CO2 sequestration': 'carbon storage',
        # "power-to-liquid": 'hydrogen derivatives',
    }

    for ptr in prefix_to_remove:
        if label[:len(ptr)] == ptr:
            label = label[len(ptr):]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for old, new in rename.items():
        if old == label:
            label = new

    for old, new in simplify.items():
        if old == label:
            label = new

    for old, new in simplify_more.items():
        if old == label:
            label = new

    return label


def rename_techs_again(label):
    simplify_more = {
        'digestible biomass': 'biomass',
        'solid biomass': 'biomass',
        'H2 pipeline': 'other',
        'H2 Electrolysis': 'other',
        'biomass': 'other biomass usage',
        'H2 Fuel Cell': 'other',
        'Li ion': 'other',
        'V2G': 'other',
        # 'carbon capture': 'other',
        # 'co2': 'other',
        # 'co2 vent': 'other',
        'helmeth': 'other',
        # 'fossil liquid fuel': 'fossil fuel + CCS'
    }

    for old, new in simplify_more.items():
        if old == label:
            label = new

    return label


# def rename_techs_transportOnly(label):
#     simplify_more = {
#         'electrofuel': 'electrofuel + CC',
#         'fossil liquid fuel': 'fossil fuel + CCS'
#     }
#
#     for old, new in simplify_more.items():
#         if old == label:
#             label = new
#
#     return label


def rename_techs_balances(label):
    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral "
    ]

    rename_if_contains = [
        "CHP",
        "gas boiler",
        "biogas",
        "solar thermal",
        "air heat pump",
        "ground heat pump",
        "resistive heater",
        "Fischer-Tropsch"
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        # "H2 Electrolysis": "hydrogen storage",
        # "H2 Fuel Cell": "hydrogen storage",
        # "H2 pipeline": "hydrogen storage",
        "battery": "battery storage",
        "CC": "CC"
    }

    rename = {
        "solar": "solar PV",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        "offwind-ac": "offshore wind (AC)",
        "offwind-dc": "offshore wind (DC)",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "co2 Store": "DAC",
        "co2 stored": "CO2 sequestration",
        "AC": "transmission lines",
        "DC": "transmission lines",
        "B2B": "transmission lines"
    }

    for ptr in prefix_to_remove:
        if label[:len(ptr)] == ptr:
            label = label[len(ptr):]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for old, new in rename.items():
        if old == label:
            label = new
    return label


preferred_order = pd.Index([
    "transmission lines",
    "hydroelectricity",
    "hydro reservoir",
    "run of river",
    "pumped hydro storage",
    "solid biomass",
    "biogas",
    "onshore wind",
    "offshore wind",
    "offshore wind (AC)",
    "offshore wind (DC)",
    "solar PV",
    "solar thermal",
    "solar",
    "building retrofitting",
    "ground heat pump",
    "air heat pump",
    "heat pump",
    "resistive heater",
    "power-to-heat",
    "gas-to-power/heat",
    "CHP",
    "OCGT",
    "gas boiler",
    "gas",
    "natural gas",
    "helmeth",
    "methanation",
    "hydrogen storage",
    "power-to-gas",
    "power-to-liquid",
    "battery storage",
    "hot water storage",
    "CO2 sequestration"
])

preferred_order2 = pd.Index([
    'power excl. fossil fuels',
    'heat',
    'CHP',
    'industry',
    'fossils',
    'H2 usages',
    'other',
    'biomass',
    'biomass to liquid',
    'other biomass usage',
    'biofuel',
    'electrofuel',
    'fossil liquid fuel',
    'opportunity cost'
])


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


def place_subplot(df, ax, ndf, position, ylabel, xlabel, title, legend=False):
    new_index = preferred_order2.intersection(df.index).append(df.index.difference(preferred_order2))
    new_columns = df.sum().index#.sort_values().index

    colors = {'power excl. fossil fuels': 'darkblue',
              'fossil fuels': 'black',
              'fossil liquid fuel': 'grey',
              'fossil fuel + CCS': 'grey',
              'fossil gas': 'darkgrey',
              'other': 'blue',
              'biomass': 'green',
              'other biomass usage': 'red',
              'biofuel': 'orange',
              'electrofuel': 'lightgreen',
              'electrofuel + CC': 'lightgreen',
              'DAC': 'pink',
              'carbon storage': 'darkgreen',
              'opportunity cost total': 'yellow',
              'opportunity cost': 'red',
              'fuel delta': 'green'}

    # colormap = plt.cm.jet
    # colors = [colormap(i) for i in np.linspace(0, 1, len(new_index))]
    # colors = sns.color_palette("hls", len(new_index))

    to_plot = df.loc[new_index,new_columns].T
    to_plot.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        #color=colors,  # [snakemake.config['plotting']['tech_colors'][i] for i in new_index]
    )

    list_values = (to_plot.astype(int).values.flatten('F'))
    print('ax patches: ', ax.patches)
    print(zip(ax.patches, list_values))
    print(list(itertools.zip_longest(ax.patches,list_values)))

    for rect, value in itertools.zip_longest(ax.patches,list_values, fillvalue=0): #zip(ax.patches, list_values):

        if ndf == 1:
            rect.set_x(rect.get_x())
        elif ndf ==2:
            rect.set_width(1 / 5)
            matrix = [-rect.get_width()*2.5, 2.5*rect.get_width()]
            rect.set_x(rect.get_x() + matrix[position] / 2)

        if value >= 6:
            h = rect.get_height() / 2.
            w = rect.get_width() / 2.
            x, y = rect.get_xy()
            ax.text(x + w, y + h, value, horizontalalignment='center', verticalalignment='center', fontsize='xx-small')


    totals = to_plot.sum(axis=1)
    # print(totals.values)

    # i = 0
    for rect, total in zip(ax.patches, totals):
        # print(total)
        # if i == 0:
        ax.text(rect.get_x() + rect.get_width() / 2, total + 20, int(total), ha='center', weight='bold')
        # if i > 0:
        #     increase = int(round((total / totals[0]-1) * 100))
        #     print(increase)
        #     ax.text(rect.get_x()+rect.get_width()/2, total+70, '+{}%'.format(increase), ha='center', fontsize='x-small')
        # i += 1

    handles, labels = ax.get_legend_handles_labels()


    handles.reverse()
    labels.reverse()

    if year == 2060:
        ymin = -30
        ymax = 400
    elif year == 2040:
        ymin = -30
        ymax = 1100

    # if transportOnly:
    ax.set_ylim([ymin, ymax])  # snakemake.config['plotting']['costs_max']])
    # else:
    #     ax.set_ylim([0, 650])  # snakemake.config['plotting']['costs_max']])

    ax.set_ylabel(ylabel)  # "System Cost [EUR billion per year]")

    ax.set_xlabel(xlabel)

    ax.set_title(title)
    ax.grid(axis='x')

    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)
    if legend == False:
        ax.get_legend().remove()


def rename_cols(df,order):
    for num in np.arange(0, 9):
        # print(df.filter(regex='0p{}'.format(str(num))).columns)
        if num == 0:
            df.columns = df.columns.str.replace('.*B0p{}Im-.*'.format(str(num)), 'Opt')
            df.columns = df.columns.str.replace('.*B0p{}ImEq-.*'.format(str(num)), '0%')
        else:
            df.columns = df.columns.str.replace('.*B0p{}.*'.format(str(num)), '{}0%'.format(num))
        df.columns = df.columns.str.replace('.*B1p0.*', '100%')

    df = df[order]
    # print(df)
    return df

# def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
#     """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
# labels is a list of the names of the dataframe, used for the legend
# title is a string for the title of the plot
# H is the hatch used for identification of the different dataframe"""
#
#     n_df = len(dfall)
#     n_col = len(dfall[1].columns)
#     n_ind = len(dfall[1].index)
#     axe = plt.subplot(111)
#
#     for df in dfall : # for each data frame
#         axe = df.plot(kind="bar",
#                       linewidth=0,
#                       stacked=True,
#                       ax=axe,
#                       legend=False,
#                       grid=False,
#                       **kwargs)  # make bar plots
#
#     h,l = axe.get_legend_handles_labels() # get the handles we want to modify
#     for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
#         for j, pa in enumerate(h[i:i+n_col]):
#             for rect in pa.patches: # for each index
#                 rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
#                 # rect.set_hatch(H * int(i / n_col)) #edited part
#                 rect.set_width(1 / float(n_df + 1))
#
#     axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
#     axe.set_xticklabels(df.index, rotation = 0)
#     axe.set_title(title)
#
#     # Add invisible data to add another legend
#     n=[]
#     for i in range(n_df):
#         n.append(axe.bar(0, 0, color="gray", hatch=H * i))
#
#     l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
#     if labels is not None:
#         l2 = plt.legend(n, labels, loc=[1.01, 0.1])
#     axe.add_artist(l1)
#     return axe


def gen_transport_df_for_plots(transportOnly, mode='cost'):

    prices = pd.read_csv(prices_dir,  # skiprows=2,
                          index_col=list(range(1)),
                          header=list(range(n_range))
                          )

    prices.columns = prices.columns.map('|'.join).str.strip('|')
    # print('prices: ', prices)

    cost_df = pd.read_csv(sdir,  # skiprows=2,
                          index_col=list(range(3)),
                          header=list(range(n_range))
                          )
    costs = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

    costs.columns = costs.columns.map('|'.join).str.strip('|')

    # convert to billions
    costs = costs / 1e9
    costs = costs.groupby(costs.index.map(rename_techs)).sum()
    costsAll = costs

    biomass = ['High', 'Med']
    if year == 2060:
        s_low = 'S400'
    elif year == 2040:
        s_low = 'S0'

    carbonstorage = [s_low, 'S1500']
    mandate = ['B0p0Im', 'B0p0ImEq', 'B0p2Im', 'B0p5Im', 'B1p0Im']

    for man, bm, cs in [(man, bm, cs) for man in mandate for bm in biomass for cs in carbonstorage]:
        sample = costs.filter(regex='{}-.*{}.*{}'.format(man, bm, cs))
        sampleRef = costs.filter(regex='B0p0Im-.*{}.*{}'.format(bm, cs))
        sampleRef.columns = sample.columns
        diff = sample.sum() - sampleRef.sum()
        diff.columns = sample.columns
        costs.loc['opportunity cost total', sample.columns] = diff[0]

    temp = costs.loc['electrofuel']
    print('Electrofuel: ', costs.loc['electrofuel'])

    if mode == 'cost':
        print('efueladdition', H2toEfuelShare * (costs.loc['H2 Electrolysis'] + costs.loc['H2 pipeline']))
        costs.loc['electrofuel'] += H2toEfuelShare * (costs.loc['H2 Electrolysis'] + costs.loc['H2 pipeline'])
        costs.loc['H2 Electrolysis'] -= H2toEfuelShare * costs.loc['H2 Electrolysis']
        costs.loc['H2 pipeline'] -= H2toEfuelShare * costs.loc['H2 pipeline']

        temp2 = costs.loc['electrofuel'] - temp

        # print('Electrofuel: ', costs.loc['electrofuel']-temp)

        if transportOnly:
            co2price = co2_shadowPrice#abs(prices.loc['co2'])
            print('co2 price', co2price)
            costs.loc['electrofuel'] += ElectrofuelAmount * 0.25714 * co2price / 1e3
            costs.loc['fossil liquid fuel'] += FossilLiquidFuelsAmount * 0.25714 * co2price / 1e3 #[TWh * MtCO2/TWh * MEUR/MtCO2 / 1000 = Billion EUR]
            costs.loc['other'] -= ElectrofuelAmount * 0.25714 * co2price / 1e3 - FossilLiquidFuelsAmount * 0.25714 * co2price / 1e3

            # Todo: Change to power incl. fossils for power production! Add CHP?
            costs.loc['electrofuel'] += H2shareofACrevenue * H2toEfuelShare * costs.loc['power excl. fossil fuels']

            print('Electrofuel: ', costs.loc['electrofuel'] - temp2)

            costs.loc['power excl. fossil fuels'] -= H2shareofACrevenue * H2toEfuelShare * costs.loc['power excl. fossil fuels']

            costs.loc['biofuel'] += SolidBiomasstoBtLshare * solid_biomass_cost#(costs.loc['solid biomass'])
            costs.loc['solid biomass'] -= SolidBiomasstoBtLshare * solid_biomass_cost#(costs.loc['solid biomass'])

    elif mode == 'price':

        costs.loc['electrofuel'] += H2toEfuelShare * H2productionTot * prices.loc['H2'] / 1e3 #[EUR/MWh = MEUR / TWh /1000 --> BEUR]
        costs.loc['electrofuel'] += ElectrofuelAmount * 0.25714 * abs(prices.loc['co2']) / 1e3

        # print('solid biomass: ', SolidBiomasstoBtLshare, SolidBiomassTot, abs(prices.loc['solid biomass']))
        # Solid biomass prices show strange behaviour!

        costs.loc['biofuel'] += SolidBiomasstoBtLshare * SolidBiomassTot * abs(prices.loc['solid biomass']) / 1e3

        costs.loc['fossil liquid fuel'] += FossilLiquidFuelsAmount * 0.25714 * abs(prices.loc['co2']) / 1e3

    costs = costs.groupby(costs.index.map(rename_techs_again)).sum()
    costsAll = costsAll.groupby(costsAll.index.map(rename_techs_again)).sum()

    to_drop = costs.index[costs.max(axis=1) < 0.5]  # snakemake.config['plotting']['costs_threshold']]
    costs = costs.drop(to_drop)

    costsAll = costsAll.drop(costsAll.index[costsAll.max(axis=1) < 0.5])

    if transportOnly:
        print('dropping non-transport')
        if year == 2060:
            to_drop = ['power excl. fossil fuels', 'other', 'other biomass usage', 'fossil gas', 'carbon storage', 'DAC']
        elif year == 2040:
            to_drop = ['power excl. fossil fuels', 'other', 'other biomass usage', 'fossil gas', 'carbon storage', 'fossil fuels']

        costs = costs.drop(to_drop, axis=0)
        costs2 = costs.drop(['opportunity cost total'], axis=0)

        for man, bm, cs in [(man, bm, cs) for man in mandate for bm in biomass for cs in carbonstorage]:
            sample = costs2.filter(regex='{}-.*{}.*{}'.format(man, bm, cs))
            sampleRef = costs2.filter(regex='B0p0Im-.*{}.*{}'.format(bm, cs))
            sampleRef.columns = sample.columns
            diff = sample.sum() - sampleRef.sum()
            diff.columns = sample.columns
            costs.loc['fuel delta', sample.columns] = diff[0]
            costs.loc['opportunity cost', sample.columns] = costs.loc['opportunity cost total', sample.columns] - diff[0]

        # costs.to_csv('../test415.csv')


        # print(costs)

        costs = costs.drop(['opportunity cost total', 'fuel delta'], axis=0)

    return costs, costsAll


def df_for_subplots(costs):
    if year == 2060:
        s_low = 'S400'
    elif year == 2040:
        s_low = 'S0'

    df2 = costs.filter(regex='High.*{}'.format(s_low))#.*B0.*Ef2.*E2.*C2.*CS2.*O2.*I0')
    df3 = costs.filter(regex='High.*S1500')#.*B0.*Ef2.*E2.*C2.*CS2.*O2.*I0')
    df4 = costs.filter(regex='Med.*{}'.format(s_low))#.*B0.*Ef2.*E2.*C2.*CS2.*O2.*I0')
    df5 = costs.filter(regex='Med.*S1500')#.*B0.*Ef2.*E2.*C2.*CS2.*O2.*I0')

    order = ['0%', 'Opt', '20%', '50%', '100%']
    df2 = rename_cols(df2, order)
    df3 = rename_cols(df3, order)
    df4 = rename_cols(df4, order)
    df5 = rename_cols(df5, order)

    return df2, df3, df4, df5


def plot_scenarios(costs, costAll, output, mode='cost'):

    df2, df3, df4, df5 = df_for_subplots(costs)
    df6, df7, df8, df9 = df_for_subplots(costAll)

    fig, ((ax2, ax3), (ax4, ax5)) = plt.subplots(2, 2, figsize=(12, 8))

    # place_subplot(df2, ax2, 'Hög biomassa, låg CO2-lagring', transportOnly)
    # place_subplot(df3, ax3, 'Hög biomassa, hög CO2-lagring', transportOnly, legend=True)
    # place_subplot(df4, ax4, 'Låg biomassa, låg CO2-lagring', transportOnly)
    # place_subplot(df5, ax5, 'Hög biomassa, hög CO2-lagring', transportOnly)

    place_subplot(df6, ax2, 2, 1, 'Fuel cost [Billion EUR]', '', 'High bio, low CS')
    place_subplot(df2, ax2, 2, 0, 'Fuel cost [Billion EUR]', '', 'High bio, low CS')

    place_subplot(df3, ax3, 1, 0, '', '', 'High bio, high CS', legend=True)
    place_subplot(df4, ax4, 1, 0, 'Fuel cost [Billion EUR]', 'Biofuel share', 'Low bio, low CS')
    place_subplot(df5, ax5, 1, 0, '', 'Biofuel share', 'High bio, high CS')

    fig.tight_layout(pad=1)

    # axes_handling_left(ax2)
    # axes_handling_left(ax4)
    # axes_handling_right(ax3,legend=True)
    # axes_handling_right(ax5)

    # if transportOnly:
    output = output + '_transportOnly_' + mode

    plt.show()

    fig.savefig(output + '.pdf', bbox_inches='tight')


#n_range = 4
# with new sensitivity wildcards:
n_range = 11

# Get shares of resources to fuel production
plot_costs(costs)
plot_balances(balances)
get_H2shareofACrevenue(H2shareofACrevenueCSV)
plot_metrics(metrics)

costs, costsAll = gen_transport_df_for_plots(transportOnly=True, mode='cost')
# prices, costAll = gen_transport_df_for_plots(transportOnly=True, mode='prices')

# plot_scenarios(output, transportOnly=False)
# plot_scenarios(output, transportOnly=False)

# plot_clustered_stacked([costs.T, costAll.T], ['df1', 'df2'])

plot_scenarios(costs, costsAll, output, mode='cost')
# plot_scenarios(prices, transportOnly=True)