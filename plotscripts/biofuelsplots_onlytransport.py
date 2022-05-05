import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.style.use('ggplot')

year = 2060
scenario = 'serverResults/electrofuels3'#.format(year)
sdir = '../results/{}/csvs/costs.csv'.format(scenario)
output = '../results/{}/plots/fuelSupply{}'.format(scenario,year)
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

            CCUS_DACshare = 0#df.loc['DAC'] / df[df > 0].dropna().sum()

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
        'uranium': 'nuclear',
        'process emissions': 'industry',
        'gas for industry': 'industry',
        'gas for mediumT industry': 'industry',
        'lowT process steam H2': 'industry',
        'SMR': 'other',
        'CC': 'carbon capture',
        'methanation': 'other',
        'nuclear': 'nuclear',
        'nuclear_new': 'nuclear',
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
    }

    for old, new in simplify_more.items():
        if old == label:
            label = new

    return label

def rename_fossil_fuels(label):
    # if year == 2040:
    #     simplify_more = {'fossil liquid fuel': 'fossil liquid fuel w/o CCS'}
    # elif year == 2060:
    #     simplify_more = {'fossil liquid fuel': 'fossil liquid fuel + CCS'}

    # for old, new in simplify_more.items():
    #     if old == label:
    #         label = new

    return label

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
    'other power',
    'solar PV',
    'onshore wind',
    'offshore wind',
    'nuclear',
    'heat',
    'CHP',
    'hydrogen',
    'industry',
    'fossils',
    'H2 usages',
    'other',
    'biomass',
    'biomass to liquid',
    'other biomass usage',
    'biofuel',
    'electrofuel',
    'fossil liquid fuel + CCS',
    'fossil liquid fuel w/o CCS',
    'fossil liquid fuel',
    'opportunity cost',
    'fossil liquid fuel emission cost',
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

colors = {'power excl. fossil fuels': '#6495ED',
              'other power': '#6495ED',#'#235ebc',
              'nuclear': 'brown',
              'offshore wind': 'darkblue',
              'onshore wind': 'lightblue',
              'solar PV': 'gold',
              'fossil fuels': '#C0C0C0',
              'fossil liquid fuel': '#C0C0C0',#'#708090',
              'fossil fuel + CCS': 'grey',
              'fossil liquid fuel + CCS': '#708090',
              'fossil liquid fuel w/o CCS': '#708090',
              'fossil gas': 'darkgrey',
              'other': 'pink',
              'hydrogen': 'white',
              'CHP': 'red',
              'biomass': 'green',
              'other biomass usage': '#FFE5B4',
              'biofuel': '#32CD32',#'#ADFF2F',#'#00FF00',#'#32CD33',#'#C2B280',
              'electrofuel': 'gold',
              'electrofuel + CC': 'lightgreen',
              'DAC': 'pink',
              'carbon storage': 'darkgreen',
              'opportunity cost total': 'blue',
              'opportunity cost': '#C04000',
              'fuel delta': 'green',
              'biomass domestic': '#40AA00',
              'biomass import': '#48CC22',
              'biofuel process': 'lightgreen',
              'electrofuel process': '#E30B5C',
              'fossil liquid fuel emission cost': '#E5E4B7',
              # 'electrofuel + CC': '#832473',  # 'lightgreen',
              # 'carbon storage': 'darkgreen'
              }

def place_subplot(df, ax, ndf, position, ylabel, xlabel, title, plottype, twolegend=False, legend=False):

    new_index = preferred_order2.intersection(df.index).append(df.index.difference(preferred_order2))
    new_columns = df.sum().index#.sort_values().index

    to_plot = df.loc[new_index,new_columns].T
    to_plot.plot(
        kind=plottype,
        ax=ax,
        stacked=True,
        color=colors,  # [snakemake.config['plotting']['tech_colors'][i] for i in new_index]
    )
    # list_values[i] = (to_plot.astype(int).values.flatten('F'))
    # i+=1

    list_values = to_plot.astype(int).values.flatten('F')
    # print(list_values)
    # print('ax patches: ', ax.patches[0:8])
    # print(list(zip(ax.patches, list_values)))
    # print(list(itertools.zip_longest(ax.patches,list_values)))

    #     h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    #     for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
    #         for j, pa in enumerate(h[i:i+n_col]):
    #             for rect in pa.patches: # for each index
    #                 rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
    #                 # rect.set_hatch(H * int(i / n_col)) #edited part
    #                 rect.set_width(1 / float(n_df + 1))

    totals = to_plot.sum(axis=1)
    totals2 = to_plot[to_plot>0].sum(axis=1)
    # print('totals: ', totals)
    # print('totals2: ', totals2)

    for rect, total in itertools.zip_longest(ax.patches, totals, fillvalue=0):  # zip(ax.patches, list_values):# #

        if ndf == 1:
            pass
        elif ndf == 2:
            rect.set_width(1 / 4)
            matrix = [-rect.get_width() / 4, 1.5 * rect.get_width()]
            rect.set_x(rect.get_x() + matrix[position])  # / 2)

    if position == ndf - 2:
        for rect, total in itertools.zip_longest(ax.patches, totals, fillvalue=0):  # zip(ax.patches, list_values):# #

            if abs(rect.get_height()) >= 20:
                h = rect.get_height() / 2.
                w = rect.get_width() / 2.
                x, y = rect.get_xy()
                ax.text(x + w, y + h, int(rect.get_height()), horizontalalignment='center', verticalalignment='center', fontsize='xx-small')

    for rect, total, total2 in itertools.zip_longest(ax.patches, totals, totals2, fillvalue=0):#zip(ax.patches, totals):#

        increase = round((total / totals[1]-1) * 100, 1)
        if abs(increase) >= 1:
            increase = int(increase)
        # print('increase: ', increase)

        if ndf == 2:
            if total > 0:
                if position == 0:
                    ax.text(rect.get_x() - rect.get_width(), total + 20, int(total), ha='center', weight='bold', fontsize='small')
                    ax.text(rect.get_x() - rect.get_width()*0.9, -40, 'Total', ha='center', va='top', fontsize='small', rotation=90, color='gray')
                # if i > 0:
                    if increase > 0:
                        insert = '+{}%'.format(increase)
                    elif increase < 0:
                        insert = '-{}%'.format(increase)
                    elif increase == 0:
                        insert = ''
                    ax.text(rect.get_x() - rect.get_width(), total2+60, insert, ha='center', fontsize='x-small')
            # i += 1

                if position == 1:
                    ax.text(rect.get_x() + rect.get_width() / 4, total2 + 20, int(total), ha='center', weight='bold', fontsize='small')
                    ax.text(rect.get_x() + rect.get_width() / 3, -40, 'Fuel', ha='center', va='top', fontsize='small', rotation=90, color='gray')
                # if i > 0:
                    if increase > 0:
                        insert = '+{}%'.format(increase)
                    elif increase < 0:
                        insert = '-{}%'.format(increase)
                    elif increase == 0:
                        insert = ''
                    ax.text(rect.get_x() + rect.get_width() / 4, total2 + 60, insert, ha='center',
                            fontsize='x-small')
        elif ndf == 1:
            if total > 0:
                ax.text(rect.get_x() + rect.get_width()/2, total2 + 20, int(total), ha='center', weight='bold',
                    fontsize='small')
                if increase > 0:
                    insert = '+{}%'.format(increase)
                elif increase < 0:
                    insert = '-{}%'.format(increase)
                elif increase == 0:
                    insert = ''
                # ax.text(rect.get_x() - rect.get_width(), total2 + 60, insert, ha='center', fontsize='x-small')


    handles, labels = ax.get_legend_handles_labels()


    handles.reverse()
    labels.reverse()

    if year == 2060:
        ymin = -30
        ymax = 650
    elif year == 2040:
        ymin = -30
        ymax = 1000

    # if transportOnly:
    ax.set_ylim([ymin, ymax])  # snakemake.config['plotting']['costs_max']])
    # else:
    #     ax.set_ylim([0, 650])  # snakemake.config['plotting']['costs_max']])

    ax.set_ylabel(ylabel)  # "System Cost [EUR billion per year]")

    ax.set_xlabel(xlabel)

    ax.set_title(title)
    ax.grid(axis='x')

    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)

    if twolegend == True:
        legend1 = ax.legend(handles[0:-5], labels[0:-5], ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)
        legend2 = ax.legend(handles[-5:], labels[-5:], ncol=1, loc="lower left", bbox_to_anchor=[1, -0.2], frameon=False)
        ax.add_artist(legend2)
        ax.add_artist(legend1)

    if legend == False:
        ax.get_legend().remove()


def rename_cols(df,order):
    print(df)
    for num in np.arange(0, 9):
        # print(df.filter(regex='0p{}'.format(str(num))).columns)
        if num == 0:
            df.columns = df.columns.str.replace('.*Ef0.*', '4000 €/kW')
        else:
            df.columns = df.columns.str.replace('.*Ef1.*', '6000 €/kW')

    df = df[order]
    print(df)
    return df


def rename_techsAll(label):

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
        "H2 Electrolysis": "hydrogen storage",
        "H2 Fuel Cell": "hydrogen storage",
        "H2 pipeline": "hydrogen storage",
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
        # "onshore wind": 'power',
        # "offshore wind": 'power',
        "offshore wind (AC)": 'offshore wind',
        "offshore wind (DC)": 'offshore wind',
        # "solar PV": 'power',
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
        "hydrogen storage": 'hydrogen',
        "power-to-gas": 'hydrogen derivatives',
        "H2": 'hydrogen',
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
        "lowT process steam solid biomass": 'other biomass usage',
        "lowT process steam methane": 'industry',
        "battery storage": 'power',
        "hot water storage": 'heat',
        'coal': 'fossil fuels',
        'oil': 'fossil fuels',
        'gas': 'fossil fuels',
        'lignite': 'fossil fuels',
        'uranium': 'power',
        'process emissions': 'industry',
        'gas for industry': 'industry',
        'lowT process steam H2': 'industry',
        'lowT process steam electricity': 'industry',
        'SMR': 'other',
        'CC': 'other',
        'methanation': 'other',
        'nuclear': 'nuclear',
        'nuclear_new': 'nuclear',
        'lowT process steam heat pump': 'industry',
        'solid biomass for mediumT industry': 'other biomass usage',
        'gas for highT industry': 'industry',
        'V2G': 'power',
        'BEV charger': 'other',
        'Li ion': 'power',
        'co2': 'other',
        'co2 vent': 'other',
        'gas for mediumT industry': 'industry',
        'helmeth': 'other',
        'sewage sludge digestible biomass': 'digestible biomass',
        'solid biomass to hydrogen': 'other biomass usage',
        'oil boiler': 'heat'
    }

    simplify_more = {
        'heat': 'other',
        'industry': 'other',
        # 'CHP': 'other',
        'power': 'other power',
        'hydrogen derivatives': 'other',
        'digestible biomass': 'biomass',
        'solid biomass': 'biomass',
        'biomass to liquid': 'biofuel process',
        'electrofuel': 'electrofuel process',
        'DAC': 'DAC',
        'CO2 sequestration': 'other',
        # "power-to-liquid": 'hydrogen derivatives',
    }

    for ptr in prefix_to_remove:
        if label[:len(ptr)] == ptr:
            label = label[len(ptr):]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old,new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for old,new in rename.items():
        if old == label:
            label = new

    for old,new in simplify.items():
        if old == label:
            label = new

    for old,new in simplify_more.items():
        if old == label:
            label = new

    return label

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
    costsAll = costs
    costs = costs.groupby(costs.index.map(rename_techs)).sum()
    costsAll = costsAll.groupby(costsAll.index.map(rename_techsAll)).sum()

    biomass = ['High', 'Med']
    if year == 2060:
        s_low = 'S400'
    elif year == 2040:
        s_low = 'S0'

    carbonstorage = [s_low, 'S1500']
    mandate = ['B0p0Im']

    for man, bm, cs in [(man, bm, cs) for man in mandate for bm in biomass for cs in carbonstorage]:
        sample = costs.filter(regex='{}-.*{}.*{}'.format(man, bm, cs))
        sampleRef = costs.filter(regex='B0p0Im-.*{}.*{}'.format(bm, cs))
        sampleRef.columns = sample.columns
        diff = sample.sum() - sampleRef.sum()
        diff.columns = sample.columns
        # print('Sample cols: ', sample.columns)
        costs.loc['opportunity cost total', sample.columns] = diff[0]

    if mode == 'cost':
        costs.loc['electrofuel'] += H2toEfuelShare * (costs.loc['H2 Electrolysis'] + costs.loc['H2 pipeline'])
        costs.loc['H2 Electrolysis'] -= H2toEfuelShare * costs.loc['H2 Electrolysis']
        costs.loc['H2 pipeline'] -= H2toEfuelShare * costs.loc['H2 pipeline']

        if transportOnly:
            co2price = abs(prices.loc['co2'])
            # print('co2 price', co2price)
            costs.loc['electrofuel'] += ElectrofuelAmount * 0.25714 * co2price / 1e3

            # if year == 2040:
            #     fossilAddition = 0 #Assumed unabated
            # elif year == 2060:
            fossilAddition = FossilLiquidFuelsAmount * 0.25714 * co2price / 1e3 #[TWh * MtCO2/TWh * MEUR/MtCO2 / 1000 = Billion EUR]

            fossilAddition.name = 'fossil liquid fuel emission cost'

            costs = costs.append(fossilAddition)
            # costs.loc['fossil liquid fuel'] += fossilAddition
            costs.loc['other'] -= ElectrofuelAmount * 0.25714 * co2price / 1e3 - fossilAddition

            # Todo: Change to power incl. fossils for power production! Add CHP?
            costs.loc['electrofuel'] += H2shareofACrevenue * H2toEfuelShare * costs.loc['power excl. fossil fuels']

            costs.loc['power excl. fossil fuels'] -= H2shareofACrevenue * H2toEfuelShare * costs.loc['power excl. fossil fuels']

            costs.loc['biofuel'] += SolidBiomasstoBtLshare * solid_biomass_cost #(costs.loc['solid biomass'])
            costs.loc['solid biomass'] -= SolidBiomasstoBtLshare * solid_biomass_cost #(costs.loc['solid biomass'])

    elif mode == 'price':

        costs.loc['electrofuel'] += H2toEfuelShare * H2productionTot * prices.loc['H2'] / 1e3 #[EUR/MWh = MEUR / TWh /1000 --> BEUR]
        costs.loc['electrofuel'] += ElectrofuelAmount * 0.25714 * abs(prices.loc['co2']) / 1e3

        # print('solid biomass: ', SolidBiomasstoBtLshare, SolidBiomassTot, abs(prices.loc['solid biomass']))
        # Solid biomass prices show strange behaviour!

        costs.loc['biofuel'] += SolidBiomasstoBtLshare * SolidBiomassTot * abs(prices.loc['solid biomass']) / 1e3

        costs.loc['fossil liquid fuel'] += FossilLiquidFuelsAmount * 0.25714 * abs(prices.loc['co2']) / 1e3

    costs = costs.groupby(costs.index.map(rename_techs_again)).sum()

    costs = costs.groupby(costs.index.map(rename_fossil_fuels)).sum()

    # costsAll = costsAll.groupby(costsAll.index.map(rename_techs_again)).sum()

    to_drop = costs.index[costs.max(axis=1) < 0.5]  # snakemake.config['plotting']['costs_threshold']]
    costs = costs.drop(to_drop)

    costsAll = costsAll.drop(costsAll.index[abs(costsAll.max(axis=1)) < 0.5])
    # costsAll = costsAll.drop(['opportunity cost total'])

    if transportOnly:
        print('dropping non-transport')
        if year == 2060:
            to_drop = ['power excl. fossil fuels', 'other', 'other biomass usage', 'fossil gas', 'carbon storage']#, 'DAC']
        elif year == 2040:
            to_drop = ['power excl. fossil fuels', 'other', 'other biomass usage', 'fossil gas', 'carbon storage', 'fossil fuels']

        costs = costs.drop(to_drop, axis=0)
        costs2 = costs#.drop(['opportunity cost total'], axis=0)

        for man, bm, cs in [(man, bm, cs) for man in mandate for bm in biomass for cs in carbonstorage]:
            sample = costs2.filter(regex='{}-.*{}.*{}'.format(man, bm, cs))
            sampleRef = costs2.filter(regex='B0p0Im-.*{}.*{}'.format(bm, cs))
            sampleRef.columns = sample.columns
            diff = sample.sum() - sampleRef.sum()
            diff.columns = sample.columns
            costs.loc['fuel delta', sample.columns] = diff[0]
            #costs.loc['opportunity cost', sample.columns] = costs.loc['opportunity cost total', sample.columns] - diff[0]

        # costs.to_csv('../test415.csv')


        # print(costsAll)

        # costs = costs.drop(['opportunity cost total', 'fuel delta'], axis=0)

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

    order = ['6000 €/kW', '4000 €/kW']
    df2 = rename_cols(df2, order)
    df3 = rename_cols(df3, order)
    df4 = rename_cols(df4, order)
    df5 = rename_cols(df5, order)

    return df2, df3, df4, df5


def plot_scenarios(costs, costAll, output, mode='cost'):

    df2, df3, df4, df5 = df_for_subplots(costs)
    df6, df7, df8, df9 = df_for_subplots(costAll)
    print(df6)

    fig, ((ax2, ax3), (ax4, ax5)) = plt.subplots(2, 2, figsize=(12, 8))

    # place_subplot(df2, ax2, 'Hög biomassa, låg CO2-lagring', transportOnly)
    # place_subplot(df3, ax3, 'Hög biomassa, hög CO2-lagring', transportOnly, legend=True)
    # place_subplot(df4, ax4, 'Låg biomassa, låg CO2-lagring', transportOnly)
    # place_subplot(df5, ax5, 'Hög biomassa, hög CO2-lagring', transportOnly)

    # place_subplot(df2, ax2, 2, 1, 'Fuel cost [Billion EUR]', '', 'High bio, low CS')
    # place_subplot(df6, ax2, 2, 0, 'Fuel cost [Billion EUR]', '', 'High bio, low CS')

    # ax2.plot(df6.sum(), linewidth=0, marker='_', ms=20, mew=2, color='black', label='total energy system cost')
    # place_subplot(df2, ax2, 1, 0, 'Fuel cost [Billion EUR]', '', 'High bio, low CS', 'bar')
    place_subplot(df6, ax2, 1, 0, 'Cost [Billion EUR]', '', 'High bio, low CS', 'bar')

    # ax3.plot(df7.sum(), linewidth=0, marker='_', ms=20, mew=2, color='black', label='total energy system cost')
    # place_subplot(df3, ax3, 1, 0, '', '', 'High bio, high CS', 'bar', legend=True)
    place_subplot(df7, ax3, 1, 0, '', '', 'High bio, high CS', 'bar', legend=True)

    # place_subplot(df4, ax4, 1, 0, 'Fuel cost [Billion EUR]', 'Biofuel share', 'Low bio, low CS', 'bar')
    place_subplot(df8, ax4, 1, 0, 'Cost [Billion EUR]', 'Nuclear cost', 'Low bio, low CS', 'bar')
    # ax4.plot(df8.sum(), linewidth=0, marker='_', ms=20, mew=2, color='black')

    # place_subplot(df5, ax5, 1, 0, '', 'Biofuel share', 'Low bio, high CS', 'bar')
    place_subplot(df9, ax5, 1, 0, '', 'Nuclear cost', 'Low bio, high CS', 'bar')
    # ax5.plot(df9.sum(), linewidth=0, marker='_', ms=20, mew=2, color='black')

    # place_subplot(df3, ax3, 2, 1, '', '', 'High bio, high CS', 'bar')
    # place_subplot(df7, ax3, 2, 0, '', '', 'High bio, high CS', 'bar', twolegend=True, legend=True)
    #
    # place_subplot(df4, ax4, 2, 1, 'Cost [Billion EUR]', 'Biofuel share', 'Low bio, low CS', 'bar')
    # place_subplot(df8, ax4, 2, 0, 'Cost [Billion EUR]', '', 'Low bio, low CS', 'bar')
    #
    # place_subplot(df5, ax5, 2, 1, '', 'Biofuel share', 'Low bio, high CS', 'bar')
    # place_subplot(df9, ax5, 2, 0, '', '', 'Low bio, high CS', 'bar')


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