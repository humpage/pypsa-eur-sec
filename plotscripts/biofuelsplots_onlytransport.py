import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

scenario = '37h_full_fixedIndustry_2060'#'serverResults/test'
sdir = '../results/{}/csvs/costs.csv'.format(scenario)
output = '../results/37h_2060_relocated'
balances = '../results/{}/csvs/supply_energy.csv'.format(scenario)


def plot_balances(balances):
    global H2toEfuelShare, ACtoElectrolysisShare, SolidBiomasstoBtLshare, FossilLiquidFuelsShare, CO2toEfuelShare, transportCO2emissionShare, TotalCO2captured, CCUS_DACshare, BiofuelCO2captureShare
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

        # print('test: ', df.index)

        # convert MWh to TWh
        df = df / 1e6

        # remove trailing link ports
        df.index = [i[:-1] if ((i != "co2") and (i[-1:] in ["0", "1", "2", "3", "4"])) else i for i in df.index]

        df = df.groupby(df.index.map(rename_techs_balances)).sum()

        # print('Dataframe: ', k, df[df < 0].dropna())#.sum())
        if k == ['H2']:
            H2toEfuelShare = df.loc['electrofuel'] / df[df < 0].dropna().sum()
            # print('Efuel: ', H2toEfuelShare)

        elif k == ['AC']:
            ACtoElectrolysisShare = df.loc['H2 Electrolysis'] / df[df < 0].dropna().sum()
            # print('Electrolysis: ', ACtoElectrolysisShare)

        elif k == ['solid biomass']:
            SolidBiomasstoBtLshare = df.loc['biomass to liquid'] / df[df < 0].dropna().sum()
            # print('BtL: ', SolidBiomasstoBtLshare)

        elif k == ['oil']:
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
           CCUS_DACshare, transportCO2emissionShare, TotalCO2captured, BiofuelCO2captureShare


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
        "lowT process steam solid biomass": 'other biomass usage',
        "lowT process steam methane": 'industry',
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
        'gas for highT industry': 'industry'
    }

    simplify_more = {
        'heat': 'other',
        'industry': 'other',
        'CHP': 'other',
        'power': 'power excl. fossils',
        'hydrogen derivatives': 'other',
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
        # 'fossil liquid fuel': 'fossil fuel + CCS'
    }

    for old, new in simplify_more.items():
        if old == label:
            label = new

    return label


def rename_techs_transportOnly(label):
    simplify_more = {
        'electrofuel': 'electrofuel + CC',
        'fossil liquid fuel': 'fossil fuel + CCS'
    }

    for old, new in simplify_more.items():
        if old == label:
            label = new

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
    'power excl. fossils',
    'heat',
    'CHP',
    'industry',
    'fossils',
    'H2 usages',
    'other',
    'biomass',
    'biomass to liquid',
    'other biomass usage',
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


def place_subplot(df, ax, ylabel, transportOnly=True, legend=False):
    new_index = preferred_order2.intersection(df.index).append(df.index.difference(preferred_order2))
    new_columns = df.sum().sort_values().index

    colors = {'power excl. fossils': 'darkblue',
              'fossils': 'black',
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
              'carbon storage': 'darkgreen'}

    # colormap = plt.cm.jet
    # colors = [colormap(i) for i in np.linspace(0, 1, len(new_index))]
    # colors = sns.color_palette("hls", len(new_index))
    print('df: ', df)
    print('DF to plot: ', df.loc[new_index,new_columns].T)
    df.loc[new_index, new_columns].T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=colors,  # [snakemake.config['plotting']['tech_colors'][i] for i in new_index]
    )

    handles, labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    if transportOnly:
        ax.set_ylim([0, 350])  # snakemake.config['plotting']['costs_max']])
    else:
        ax.set_ylim([0, 650])  # snakemake.config['plotting']['costs_max']])

    ax.set_ylabel(ylabel)  # "System Cost [EUR billion per year]")

    ax.set_xlabel("Biofuel mandate")

    ax.grid(axis='x')

    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)
    if legend == False:
        ax.get_legend().remove()


def rename_cols(df):
    for num in np.arange(0, 9):
        # print(df.filter(regex='0p{}'.format(str(num))).columns)
        if num == 0:
            df.columns = df.columns.str.replace('.*B0p{}.*'.format(str(num)), 'Opt')
        else:
            df.columns = df.columns.str.replace('.*B0p{}.*'.format(str(num)), '>{}0%'.format(num))
        df.columns = df.columns.str.replace('.*B1p0.*', '100%')

    # df.columns = df.columns.str.replace('.*-H-.*'.format(str(num)), 'NoBio')
    # print(df.columns)


def plot_scenarios(output, transportOnly):
    cost_df = pd.read_csv(sdir,  # skiprows=2,
                          index_col=list(range(3)),
                          header=list(range(n_range))
                          )

    df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

    df.columns = df.columns.map('|'.join).str.strip('|')

    # convert to billions
    df = df / 1e9

    df = df.groupby(df.index.map(rename_techs)).sum()

    # H2toEfuelShare, ACtoElectrolysisShare, SolidBiomasstoBtLshare, FossilLiquidFuelsShare
    # CO2toEfuelShare
    # print('Tesdkujhsdg: ',df)

    df.loc['electrofuel'] += H2toEfuelShare * (df.loc['H2 Electrolysis'] + df.loc['H2 pipeline'])
    df.loc['H2 Electrolysis'] -= H2toEfuelShare * df.loc['H2 Electrolysis']
    df.loc['H2 pipeline'] -= H2toEfuelShare * df.loc['H2 pipeline']

    # CCUS_DACshare, transportCO2emissionShare, TotalCO2captured

    if transportOnly:
        print('Respreading CCUS cost to transport')
        # Beware: can only be used when looking at transport only, since the cost is not factored out of the CO2 sources!
        CHP_capture_cost = 26.4 / 1000  # B€/MtCO2
        df.loc['electrofuel'] += CO2toEfuelShare * (
                    df.loc['DAC'] + (1 - CCUS_DACshare) * TotalCO2captured * CHP_capture_cost)
        liquidFuelAddition = transportCO2emissionShare * FossilLiquidFuelsShare * \
                             (df.loc['DAC'] + (1 - CCUS_DACshare) * TotalCO2captured * CHP_capture_cost + df.loc[
                                 'carbon storage'])
        # print('Liquid fuel addition: ', liquidFuelAddition)
        df.loc['fossil liquid fuel'] += liquidFuelAddition

        #TODO: decrease biofuel cost with the carbon capture cost
        # print(df.loc['biofuel'])
        # df.loc['biofuel'] -= (CO2toEfuelShare + transportCO2emissionShare * FossilLiquidFuelsShare)\
        #                      * BiofuelCO2captureShare * (1 - CCUS_DACshare) * TotalCO2captured * CHP_capture_cost
        # print(df.loc['biofuel'])

    # Todo: Change to power incl. fossils for power production!
    electrolysis_elec_price_weight = 0.4
    df.loc['electrofuel'] += electrolysis_elec_price_weight * ACtoElectrolysisShare * H2toEfuelShare * (df.loc['power excl. fossils'])
    df.loc['power excl. fossils'] -= electrolysis_elec_price_weight * ACtoElectrolysisShare * H2toEfuelShare * (df.loc['power excl. fossils'])

    df.loc['biofuel'] += SolidBiomasstoBtLshare * (df.loc['solid biomass'])
    df.loc['solid biomass'] -= SolidBiomasstoBtLshare * (df.loc['solid biomass'])

    # print('Testafter: ', df)

    df = df.groupby(df.index.map(rename_techs_again)).sum()
    if transportOnly:
        df = df.groupby(df.index.map(rename_techs_transportOnly)).sum()

    to_drop = df.index[df.max(axis=1) < 0.5]  # snakemake.config['plotting']['costs_threshold']]

    # print("dropping")
    #
    # print(df.loc[to_drop])

    df = df.drop(to_drop)

    if transportOnly:
        print('dropping non-transport')
        df = df.drop(['power excl. fossils', 'other', 'other biomass usage', 'fossil gas', 'carbon storage', 'DAC'],
                     axis=0)
        # return df

    print('DF SUM', df.sum())
    df2 = df.filter(regex='High.*S400')#.*B0.*Ef2.*E2.*C2.*CS2.*O2.*I0')
    df3 = df.filter(regex='High.*S1500')#.*B0.*Ef2.*E2.*C2.*CS2.*O2.*I0')
    df4 = df.filter(regex='Med.*S400')#.*B0.*Ef2.*E2.*C2.*CS2.*O2.*I0')
    df5 = df.filter(regex='Med.*S1500')#.*B0.*Ef2.*E2.*C2.*CS2.*O2.*I0')

    rename_cols(df2)
    rename_cols(df3)
    rename_cols(df4)
    rename_cols(df5)

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

    plt.show()

    fig.savefig(output + '.pdf', bbox_inches='tight')


n_range = 4
# with new sensitivity wildcards:
# n_range = 11

# Get shares of resources to fuel production
plot_balances(balances)

plot_scenarios(output, transportOnly=False)
plot_scenarios(output, transportOnly=False)
plot_scenarios(output, transportOnly=True)