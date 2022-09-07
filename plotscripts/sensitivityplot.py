import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["r", "#e94cdc", "b"])
#['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold']

plt.style.use('ggplot')

scenario2060 = 'serverResults/sensitivities2060_final'
scenario2040 = 'serverResults/sensitivities2040_final'
sdir = '../results/{}/csvs/costs.csv'.format(scenario2060)
output = '../results/sensitivities2060_violin'
balances = '../results/{}/csvs/supply_energy.csv'.format(scenario2060)
metrics2040 = '../results/{}/csvs/metrics.csv'.format(scenario2040)
metrics2060 = '../results/{}/csvs/metrics.csv'.format(scenario2060)

scenarioMain = 'serverResults/mainScenarios'
sdir2060 = '../results/serverResults/mainScenarios2060_final/csvs/costs.csv'
sdir2040 = '../results/serverResults/mainScenarios2040_final/csvs/costs.csv'
balances2040 = '../results/{}2040_final/csvs/supply_energy.csv'.format(scenarioMain)
balances2060 = '../results/{}2060_final/csvs/supply_energy.csv'.format(scenarioMain)


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
        "solid biomass import": 'biomass import',
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
        'nuclear': 'power',
        'nuclear_new': 'power',
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
        'CHP': 'other',
        'power': 'power excl. fossils',
        'hydrogen derivatives': 'other',
        'digestible biomass': 'biomass domestic',
        'solid biomass': 'biomass domestic',
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

def rename_cols(df,order):
    for num in np.arange(0, 9):
        if num == 0:
            df.columns = df.columns.str.replace('.*B0p{}Im-.*'.format(str(num)), 'Opt')
            df.columns = df.columns.str.replace('.*B0p{}ImEq-.*'.format(str(num)), '0%')
        else:
            df.columns = df.columns.str.replace('.*B0p{}.*'.format(str(num)), '{}0%'.format(num))
        df.columns = df.columns.str.replace('.*B1p0.*', '100%')

    df = df[order]
    return df


def plot_balances(balances):
    global H2toEfuelShare, ACtoElectrolysisShare, SolidBiomasstoBtLshare, FossilLiquidFuelsShare, CO2toEfuelShare,\
        transportCO2emissionShare, TotalCO2captured, CCUS_DACshare, BiofuelCO2captureShare, BtLshare

    co2_carriers = ["co2", "co2 stored", "process emissions"]

    balances_df = pd.read_csv(
        balances,
        index_col=list(range(3)),
        header=list(range(11))
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
            BtLshare = abs(df.loc['biomass to liquid'] / df[df < 0].dropna().sum())
            # print('Fossil fuels share: ', FossilLiquidFuelsShare)
            # print('BtL share: ', BtLshare)

        elif k == ['co2 stored']:
            CO2toEfuelShare = df.loc['electrofuel'] / df[df < 0].dropna().sum()
            # print('CO2 to Efuel share: ', CO2toEfuelShare)

            BiofuelCO2captureShare = df.loc['biomass to liquid'] / df[df > 0].dropna().sum()
            # print('Biofuel captured CO2 share: ', BiofuelCO2captureShare)

            #CCUS_DACshare = df.loc['DAC'] / df[df > 0].dropna().sum()
            # print('CCUS DAC share: ', CCUS_DACshare)

            TotalCO2captured = df[df > 0].sum()
            # print('Total CO2 captured: ', TotalCO2captured, ' MtCO2')

        elif k == ['co2']:
            transportCO2emissionShare = (df.loc['electrofuel'] + df.loc['biomass to liquid'] + df.loc['oil emissions'] +
                                         df.loc['shipping oil emissions']) / df[df > 0].dropna().sum()
            # print('Transport share of total CO2 emissions: ', transportCO2emissionShare)

    return H2toEfuelShare, ACtoElectrolysisShare, SolidBiomasstoBtLshare, FossilLiquidFuelsShare, CO2toEfuelShare,\
           transportCO2emissionShare, TotalCO2captured, BiofuelCO2captureShare, BtLshare#, CCUS_DACshare

def plot_balancesMain(balances):
    global BtLshare #H2toEfuelShare, ACtoElectrolysisShare, SolidBiomasstoBtLshare, FossilLiquidFuelsShare, CO2toEfuelShare,\
        #transportCO2emissionShare, TotalCO2captured, CCUS_DACshare, BiofuelCO2captureShare,

    co2_carriers = ["co2", "co2 stored", "process emissions"]

    balances_df = pd.read_csv(
        balances,
        index_col=list(range(3)),
        header=list(range(11))
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
        # print(df)
        # remove trailing link ports
        df.index = [i[:-1] if ((i != "co2") and (i[-1:] in ["0", "1", "2", "3", "4"])) else i for i in df.index]
        # print(df)
        df = df.groupby(df.index.map(rename_techs_balances)).sum()

        # print('Dataframe: ', k, df[df < 0].dropna())#.sum())
        # if k == ['H2']:
        #     H2toEfuelShare = df.loc['electrofuel'] / df[df < 0].dropna().sum()
        #     # print('Efuel: ', H2toEfuelShare)
        #
        # elif k == ['AC']:
        #     ACtoElectrolysisShare = df.loc['H2 Electrolysis'] / df[df < 0].dropna().sum()
        #     # print('Electrolysis: ', ACtoElectrolysisShare)
        #
        # elif k == ['solid biomass']:
        #     SolidBiomasstoBtLshare = df.loc['biomass to liquid'] / df[df < 0].dropna().sum()
        #     # print('BtL: ', SolidBiomasstoBtLshare)

        if k == ['oil']:
            FossilLiquidFuelsShare = abs(df.loc['oil'] / df[df < 0].dropna().sum())
            BtLshare = abs(df.loc['biomass to liquid'] / df[df < 0].dropna().sum())
            # print('Fossil fuels share: ', FossilLiquidFuelsShare)
            # print('BtL share: ', BtLshare)

        # elif k == ['co2 stored']:
        #     CO2toEfuelShare = df.loc['electrofuel'] / df[df < 0].dropna().sum()
        #     # print('CO2 to Efuel share: ', CO2toEfuelShare)
        #
        #     BiofuelCO2captureShare = df.loc['biomass to liquid'] / df[df > 0].dropna().sum()
        #     # print('Biofuel captured CO2 share: ', BiofuelCO2captureShare)
        #
        #     CCUS_DACshare = df.loc['DAC'] / df[df > 0].dropna().sum()
        #     # print('CCUS DAC share: ', CCUS_DACshare)
        #
        #     TotalCO2captured = df[df > 0].sum()
        #     # print('Total CO2 captured: ', TotalCO2captured, ' MtCO2')
        #
        # elif k == ['co2']:
        #     transportCO2emissionShare = (df.loc['electrofuel'] + df.loc['biomass to liquid'] + df.loc['oil emissions'] +
        #                                  df.loc['shipping oil emissions']) / df[df > 0].dropna().sum()
        #     # print('Transport share of total CO2 emissions: ', transportCO2emissionShare)

    return BtLshare #H2toEfuelShare, ACtoElectrolysisShare, SolidBiomasstoBtLshare, FossilLiquidFuelsShare, CO2toEfuelShare,\
           #CCUS_DACshare, transportCO2emissionShare, TotalCO2captured, BiofuelCO2captureShare,


def cost_dataframe(dir):
    cost_df = pd.read_csv(dir,  # skiprows=2,
                          index_col=list(range(3)),
                          header=list(range(n_range))
                          )

    #print(cost_df)


    df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

    df.columns = df.columns.map('|'.join).str.strip('|')
    # print(df)

    # convert to billions
    df = df / 1e9

    df = df.groupby(df.index.map(rename_techs)).sum()

    return df


def get_metrics(metrics):

    metrics_df = pd.read_csv(
        metrics,
        index_col=list(range(1)),
        header=list(range(11))
    )

    # metrics = metrics_df / 1e9

    return metrics_df

def set_axis_style(ax, labels, ylabels=False):
    ax.yaxis.set_tick_params(direction='out')
    # ax.yaxis.set_ticks_position('left')
    ax.set_yticks(np.arange(1, len(labels) + 1))#, labels=labels)
    # if ylabels:
    ax.set_yticklabels(labels)
    # elif not ylabels:
        # ax.set_yticklabels([])
    ax.set_ylim(0.25, len(labels) + 0.75)
#     ax.set_xlabel('Cost difference')
    ax.set_xlim([-1, 60])


def set_axis_style_sorted(ax, labels, ylabels=False):
    # ax.yaxis.set_tick_params(direction='out')
    # ax.yaxis.set_ticks_position('left')
    # ax.set_yticks(np.arange(1, len(labels) + 1))#, labels=labels)
    # if ylabels:
    # ax.set_yticklabels(labels)
    # elif not ylabels:
    ax.xaxis.set_ticklabels([])
    ax.axes.xaxis.set_visible(False)
    # ax.set_ylim(0.25, len(labels) + 0.75)
#    ax.set_ylabel('Parameter')
    ax.set_ylabel('Total cost increase' '\n' 'relative to base scenario [%]')
    # ax.set_xlim([0, 25])

def sample_gen(metrics,year):
    metrics_df = get_metrics(metrics)

    # vars = ['FT0', 'FT2', 'E0', 'E2', 'CC0', 'CC2', 'CS0', 'CS2', 'O0', 'O2', 'I0', 'I2']
    # mandate = ['B0p0', 'B0p2', 'B0p5']
    biomass = ['High', 'Med']

    if year == 2060:
        carbonstorage = ['S400', 'S1500']
    elif year == 2040:
        carbonstorage = ['S0', 'S1500']

    mandate = ['B0p0Im', 'B0p0ImEq', 'B0p2Im', 'B0p5Im', 'B1p0Im']

    sampledict = {}
    sampledictDiff = {}
    samplesAll = {}

# for bm, cs in [(bm,cs) for bm in biomass for cs in carbonstorage]:
#     sample = metrics_df.xs('37H-T-H-B0p5Im-{}-I-{}-CCL-solar+p3'.format(bm,cs), level='opt', axis=1)
#     sampleRef = metrics_df.xs('37H-T-H-B0p0Im-{}-I-{}-CCL-solar+p3'.format(bm,cs), level='opt', axis=1)
#     sampledict[bm,cs] = sample.loc['total costs'] / 1e9 #(sample.loc['total costs'] - sampleRef.loc['total costs']) / 1e9 # / sampleRef.loc['total costs']) * 100
    # sampledictDiff[bm,cs] = (sample.loc['total costs'] - sampleRef.loc['total costs']) / 1e9 # / sampleRef.loc['total costs']) * 100

# sample_df = pd.DataFrame.from_dict(sampledict)

    for man, bm, cs in [(man, bm, cs) for man in mandate for bm in biomass for cs in carbonstorage]:
        # if (man == 'B1p0Im' and cs == 'S400'):
        #     sample = metrics_df.xs('37H-T-H-{}-{}-I-S420-CCL-solar+p3'.format(man,bm), level='opt', axis=1)
        # else:
        sample = metrics_df.xs('37H-T-H-{}-{}-I-{}-CCL-solar+p3'.format(man,bm,cs), level='opt', axis=1)
        sampleRef = metrics_df.xs('37H-T-H-B0p0Im-{}-I-{}-CCL-solar+p3'.format(bm,cs), level='opt', axis=1)
        samplesAll[man,bm,cs] = sample.loc['total costs'] / 1e9
        sampledictDiff[man,bm,cs] = (sample.loc['total costs'] - sampleRef.loc['total costs']) / 1e9 # / sampleRef.loc['total costs']) * 100

    sample_df = pd.DataFrame.from_dict(samplesAll)

    sample_dfDiff = pd.DataFrame.from_dict(sampledictDiff)

    return sample_df, sample_dfDiff


def datagrouping(man, biomass,cs, sample_df):
    data = [100 * (sample_df.xs('FT2', level='biofuel_sensitivity')[(man, biomass,cs)] - sample_df.xs('FT0', level='biofuel_sensitivity')[(man, biomass, cs)])/sample_df.xs('FT0', level='biofuel_sensitivity')[(man, biomass, cs)],
         100 * (sample_df.xs('E2', level='electrolysis_sensitivity')[(man, biomass, cs)] - sample_df.xs('E0', level='electrolysis_sensitivity')[(man, biomass, cs)])/sample_df.xs('E0', level='electrolysis_sensitivity')[(man, biomass, cs)],
         100 * (sample_df.xs('CC2', level='cc_sensitivity')[(man, biomass, cs)] - sample_df.xs('CC0', level='cc_sensitivity')[(man, biomass, cs)])/sample_df.xs('CC0', level='cc_sensitivity')[(man, biomass, cs)],
         100 * (sample_df.xs('CS2', level='cs_sensitivity')[(man, biomass, cs)] - sample_df.xs('CS0', level='cs_sensitivity')[(man, biomass, cs)])/sample_df.xs('CS0', level='cs_sensitivity')[(man, biomass, cs)],
         100 * (sample_df.xs('O2', level='oil_sensitivity')[(man, biomass, cs)] - sample_df.xs('O0', level='oil_sensitivity')[(man, biomass, cs)])/sample_df.xs('O0', level='oil_sensitivity')[(man, biomass, cs)],
         100 * (sample_df.xs('I2', level='biomass_import_sensitivity')[(man, biomass, cs)] - sample_df.xs('I0', level='biomass_import_sensitivity')[(man, biomass, cs)])/sample_df.xs('I0', level='biomass_import_sensitivity')[(man, biomass, cs)]]
    return data


def plot_sensitivityVars(sample_df, ax1, ax2, ax3, ax4, year):

    if year == 2060:
        s_low = 'S400'
        s_low2 = 'S400'
    elif year == 2040:
        s_low = 'S0'
        s_low2 = 'S0'

    data1 = datagrouping('B0p0Im', 'High',s_low, sample_df)
    data2 = datagrouping('B0p0Im', 'Med',s_low, sample_df)
    data3 = datagrouping('B0p0Im', 'High','S1500', sample_df)
    data4 = datagrouping('B0p0Im', 'Med','S1500', sample_df)

    data5 = datagrouping('B0p2Im', 'High',s_low, sample_df)
    data6 = datagrouping('B0p2Im', 'Med',s_low, sample_df)
    data7 = datagrouping('B0p2Im', 'High','S1500', sample_df)
    data8 = datagrouping('B0p2Im', 'Med','S1500', sample_df)

    data9 = datagrouping('B0p5Im', 'High',s_low, sample_df)
    data10 = datagrouping('B0p5Im', 'Med',s_low, sample_df)
    data11 = datagrouping('B0p5Im', 'High','S1500', sample_df)
    data12 = datagrouping('B0p5Im', 'Med','S1500', sample_df)

    data13 = datagrouping('B1p0Im', 'High',s_low2, sample_df)
    data14 = datagrouping('B1p0Im', 'Med',s_low2, sample_df)
    data15 = datagrouping('B1p0Im', 'High','S1500', sample_df)
    data16 = datagrouping('B1p0Im', 'Med','S1500', sample_df)

    # bp = ax1.violinplot(data1, vert=0)
    # bp2 = ax2.violinplot(data2, vert=0)
    # bp3 = ax3.violinplot(data3, vert=0)
    # bp4 = ax4.violinplot(data4, vert=0)
    # bp5 = ax5.violinplot(data5, vert=0)
    # bp6 = ax6.violinplot(data6, vert=0)
    # bp7 = ax7.violinplot(data7, vert=0)
    # bp8 = ax8.violinplot(data8, vert=0)
    # bp9 = ax9.violinplot(data9, vert=0)
    # bp10 = ax10.violinplot(data10, vert=0)
    # bp11 = ax11.violinplot(data11, vert=0)
    # bp12 = ax12.violinplot(data12, vert=0)
    # bp13 = ax13.violinplot(data13, vert=0)
    # bp14 = ax14.violinplot(data14, vert=0)
    # bp15 = ax15.violinplot(data15, vert=0)
    # bp16 = ax16.violinplot(data16, vert=0)

    widthViolin=0.4
    bp = ax1.violinplot(data1, vert=0, widths=widthViolin, positions=[1.1, 2.1,3.1,4.1,5.1,6.1])
    bp2 = ax2.violinplot(data2, vert=0, widths=widthViolin, positions=[1.1, 2.1,3.1,4.1,5.1,6.1])
    bp3 = ax3.violinplot(data3, vert=0, widths=widthViolin, positions=[1.1, 2.1,3.1,4.1,5.1,6.1])
    bp4 = ax4.violinplot(data4, vert=0, widths=widthViolin, positions=[1.1, 2.1,3.1,4.1,5.1,6.1])
    bp5 = ax1.violinplot(data5, vert=0, widths=widthViolin, positions=[0.9, 1.9,2.9,3.9,4.9,5.9])
    bp6 = ax2.violinplot(data6, vert=0, widths=widthViolin, positions=[0.9, 1.9,2.9,3.9,4.9,5.9])
    bp7 = ax3.violinplot(data7, vert=0, widths=widthViolin, positions=[0.9, 1.9,2.9,3.9,4.9,5.9])
    bp8 = ax4.violinplot(data8, vert=0, widths=widthViolin, positions=[0.9, 1.9,2.9,3.9,4.9,5.9])
    bp9 = ax1.violinplot(data9, vert=0, widths=widthViolin, positions=[0.8, 1.8,2.8,3.8,4.8,5.8])
    bp10 = ax2.violinplot(data10, vert=0, widths=widthViolin, positions=[0.8, 1.8,2.8,3.8,4.8,5.8])
    bp11 = ax3.violinplot(data11, vert=0, widths=widthViolin, positions=[0.8, 1.8,2.8,3.8,4.8,5.8])
    bp12 = ax4.violinplot(data12, vert=0, widths=widthViolin, positions=[0.8, 1.8,2.8,3.8,4.8,5.8])
    bp13 = ax1.violinplot(data13, vert=0, widths=widthViolin)
    bp14 = ax2.violinplot(data14, vert=0, widths=widthViolin)
    bp15 = ax3.violinplot(data15, vert=0, widths=widthViolin)
    bp16 = ax4.violinplot(data16, vert=0, widths=widthViolin)

    color = 'orange'
    for v in [bp13, bp14, bp15, bp16]:
        for patch in v['bodies']:
            patch.set_color(color)
            # patch.set_edgecolor(color)
            v['cmaxes'].set_color(color)
            v['cmins'].set_color(color)
            v['cbars'].set_color(color)

    return bp4, bp8, bp12, bp16


def errorbars(bm,cs,mandate,sample_df):#,main_value):
    errorMin = []
    errorMax = []
    for i in mandate:
        errorMin.append(min(sample_df.loc[:, (i, bm, cs)]))
        errorMax.append(max(sample_df.loc[:, (i, bm, cs)]))
        # errorMin.append(main_value[mandate.index(i)] - min(sample_df.loc[:, (i, bm, cs)]))
        # errorMax.append(max(sample_df.loc[:, (i, bm, cs)]) - main_value[mandate.index(i)])

    error = [errorMin,errorMax]
    print(error)

    return error

# fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16)) = plt.subplots(nrows=4, ncols=4, figsize=(13, 17), sharey=True)

fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4, figsize=(17, 7), sharey=True)


plot_balances(balances)
sample_df2040, sample_dfDiff2040 = sample_gen(metrics2040,2040)
sample_df2060, sample_dfDiff2060 = sample_gen(metrics2060,2060)
# bp4 = plot_sensitivityVars(sample_df2040)
bp4, bp8, bp12, bp16 = plot_sensitivityVars(sample_df2060, ax5, ax6, ax7, ax8, 2060)
bp4_2, bp8_2, bp12_2, bp16_2 = plot_sensitivityVars(sample_df2040, ax1, ax2, ax3, ax4, 2040)

ax4.legend([bp4['bodies'][0], bp8['bodies'][0], bp12['bodies'][0], bp16['bodies'][0]], ['Opt', '20%', '50%', '100%'], loc="upper left", bbox_to_anchor=[1,1], frameon=False)
# handles, labels = ax4.get_legend_handles_labels()
# print(handles, labels)

label = ['Fischer-Tropsch','Electrolysis','Carbon capture','Carbon storage','Fossil price','Biomass import price']
for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:#, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16]:
    set_axis_style(ax, label)

fig.text(0.5, 0.01, 'Total system cost difference between pessimistic and optimistic parameter values [%]', ha='center')

ax1.set_title('High bio, low CS')
ax2.set_title('Low bio, low CS')
ax3.set_title('High bio, high CS')
ax4.set_title('Low bio, high CS')

ax1.set_ylabel('2040')
ax5.set_ylabel('2060')
# ax9.set_ylabel('50%')
# ax13.set_ylabel('100%')



plt.show()
fig.savefig(output + '.pdf', bbox_inches='tight')



# cost increase plot
n_range = 11
df2060 = cost_dataframe(sdir2060)
df2040 = cost_dataframe(sdir2040)


BtLshare2060 = plot_balancesMain(balances2060)
BtLshare2040 = plot_balancesMain(balances2040)

df2 = df2060.filter(regex='High.*S400')
df3 = df2060.filter(regex='High.*S1500')
df4 = df2060.filter(regex='Med.*S400')
df5 = df2060.filter(regex='Med.*S1500')

df6 = df2040.filter(regex='High.*S0')
df7 = df2040.filter(regex='High.*S1500')
df8 = df2040.filter(regex='Med.*S0')
# print(df8)
df9 = df2040.filter(regex='Med.*S1500')

order=['0%','Opt','20%','50%','100%']
df2 = rename_cols(df2, order)
df3 = rename_cols(df3, order)
df4 = rename_cols(df4, order)
df5 = rename_cols(df5, order)

df6 = rename_cols(df6, order)
df7 = rename_cols(df7, order)
df8 = rename_cols(df8, order)
df9 = rename_cols(df9, order)

fig3, ((ax98_2,ax99_2),(ax98,ax99)) = plt.subplots(2,2,figsize=(12,5), gridspec_kw={'height_ratios': [1, 8]})

# print('DF4: ',df4)
# print(df4['Opt'].sum())
# ax98.plot([0,BtLshare2040.filter(regex='B0p0Im-High.*S0')[0]*100,20,50,100],(df6.sum().values - df6['Opt'].sum()), label = 'High bio, low CS', linewidth = 1.5, color='#E30B5C')
# ax98.plot([0,BtLshare2040.filter(regex='B0p0Im-High.*S1500')[0]*100,20,50,100],(df7.sum().values - df7['Opt'].sum()), label = 'High bio, high CS', linewidth = 1.5, color='#6495ED')
# ax98.plot([0,BtLshare2040.filter(regex='B0p0Im-Med.*S0')[0]*100,20,50,100],(df8.sum().values - df8['Opt'].sum()), label = 'Low bio, low CS', linewidth = 1.5, color='#C2B280')
# ax98.plot([0,BtLshare2040.filter(regex='B0p0Im-Med.*S1500')[0]*100,20,50,100],(df9.sum().values - df9['Opt'].sum()), label = 'Low bio, high CS', linewidth = 1.5, color='gold')


errorlinewidth = 1.5
mandate = ['B0p0ImEq','B0p0Im', 'B0p2Im', 'B0p5Im', 'B1p0Im']
mandateTemp = ['B0p0ImEq', 'B0p0Im', 'B0p2Im', 'B0p5Im']

# print('sample_dfDiff: ', sample_dfDiff)
#
# print(sample_dfDiff[sample_dfDiff>1000])
sample_dfDiff2040.to_csv('../temp13247.csv')

# print('fullsample: ', sample_dfDiff2040)
# print('main value: ', df5.abs().sum())
# print('error: ', errorbars('Med','S0',mandate,sample_dfDiff2040))
# print('error-main: ', errorbars('Med','S0',mandate,sample_dfDiff2040)-df4.abs().sum().values)
#
# Error bar
# ax98.errorbar([0,BtLshare2040.filter(regex='B0p0Im-High.*S0')[0]*100,20,50,100],(df6.abs().sum().values - df6['Opt'].sum()),
#               label = 'High bio, low CS', yerr=abs(errorbars('High','S0',mandate,sample_dfDiff2040)-(df6.abs().sum().values - df6['Opt'].sum())), elinewidth=errorlinewidth, linewidth = 1.5, color='#E30B5C')
# ax98.errorbar([0+a,BtLshare2040.filter(regex='B0p0Im-High.*S1500')[0]*100+a,20+a,50+a,100+a],(df7.abs().sum().values - df7['Opt'].sum()),
#           label = 'High bio, high CS', yerr=abs(errorbars('High','S1500',mandate,sample_dfDiff2040)-(df7.abs().sum().values - df7['Opt'].sum())), elinewidth=errorlinewidth, linewidth = 1.5, color='#6495ED')
# ax98.errorbar([0-a,BtLshare2040.filter(regex='B0p0Im-Med.*S0')[0]*100-a,20-a,50-a,100-a],(df8.abs().sum().values - df8['Opt'].sum()),
#           label = 'Low bio, low CS', yerr=abs(errorbars('Med','S0',mandate,sample_dfDiff2040)-(df8.abs().sum().values - df8['Opt'].sum())), elinewidth=errorlinewidth, linewidth = 1.5, color='#C2B280')
# ax98.errorbar([0-2*a,BtLshare2040.filter(regex='B0p0Im-Med.*S1500')[0]*100-2*a,20-2*a,50-2*a,100-2*a],(df9.abs().sum().values - df9['Opt'].sum()),
#           label = 'Low bio, high CS', yerr=abs(errorbars('Med','S1500',mandate,sample_dfDiff2040)-(df9.abs().sum().values - df9['Opt'].sum())), elinewidth=errorlinewidth, linewidth = 1.5, color='gold')

a = 0
ax98.plot([0,BtLshare2040.filter(regex='B0p0Im-High.*S0')[0]*100,20,50,100],(df6.abs().sum().values - df6['Opt'].sum()), label = 'High bio, low CS', color='#E30B5C')
ax98.plot([0+a,BtLshare2040.filter(regex='B0p0Im-High.*S1500')[0]*100+a,20+a,50+a,100+a],(df7.abs().sum().values - df7['Opt'].sum()), label = 'High bio, high CS', color='#6495ED')
ax98.plot([0-a,BtLshare2040.filter(regex='B0p0Im-Med.*S0')[0]*100-a,20-a,50-a,100-a],(df8.abs().sum().values - df8['Opt'].sum()), label = 'Low bio, low CS', color='#C2B280')
ax98.plot([0-2*a,BtLshare2040.filter(regex='B0p0Im-Med.*S1500')[0]*100-2*a,20-2*a,50-2*a,100-2*a],(df9.abs().sum().values - df9['Opt'].sum()), label = 'Low bio, high CS', color='gold')

#data0p5 = [sample_dfDiff2040.loc[:, ('B0p5Im', 'High', 'S0')], sample_dfDiff2040.loc[:, ('B0p5Im', 'Med', 'S0')], sample_dfDiff2040.loc[:, ('B0p5Im', 'High', 'S1500')], sample_dfDiff2040.loc[:, ('B0p5Im', 'Med', 'S1500')]]
data1 = [sample_dfDiff2040.loc[:, ('B0p0ImEq', 'High', 'S0')], sample_dfDiff2040.loc[:, ('B0p2Im', 'High', 'S0')], sample_dfDiff2040.loc[:, ('B0p5Im', 'High', 'S0')], sample_dfDiff2040.loc[:, ('B1p0Im', 'High', 'S0')]]
data2 = [sample_dfDiff2040.loc[:, ('B0p0ImEq', 'High', 'S1500')], sample_dfDiff2040.loc[:, ('B0p2Im', 'High', 'S1500')], sample_dfDiff2040.loc[:, ('B0p5Im', 'High', 'S1500')], sample_dfDiff2040.loc[:, ('B1p0Im', 'High', 'S1500')]]
data3 = [sample_dfDiff2040.loc[:, ('B0p0ImEq', 'Med', 'S0')], sample_dfDiff2040.loc[:, ('B0p2Im', 'Med', 'S0')], sample_dfDiff2040.loc[:, ('B0p5Im', 'Med', 'S0')], sample_dfDiff2040.loc[:, ('B1p0Im', 'Med', 'S0')]]
data4 = [sample_dfDiff2040.loc[:, ('B0p0ImEq', 'Med', 'S1500')], sample_dfDiff2040.loc[:, ('B0p2Im', 'Med', 'S1500')], sample_dfDiff2040.loc[:, ('B0p5Im', 'Med', 'S1500')], sample_dfDiff2040.loc[:, ('B1p0Im', 'Med', 'S1500')]]
#vplot = ax98.violinplot(data0p5, widths=3, positions=[47,49,51,53])
a=1.5
w=3

#Violin plots
vplot1 = ax98.violinplot(data1, widths=w, positions=[0-3*a,20-3*a,50-3*a,100-3*a])
vplot2 = ax98.violinplot(data2, widths=w, positions=[0-a,20-a,50-a,100-a])
vplot3 = ax98.violinplot(data3, widths=w, positions=[0+a,20+a,50+a,100+a])
vplot4 = ax98.violinplot(data4, widths=w, positions=[0+3*a,20+3*a,50+3*a,100+3*a])

#ax98.violinplot([sample_dfDiff2040.loc[:, ('B0p2Im', 'High', 'S0')], sample_dfDiff2040.loc[:, ('B0p2Im', 'Med', 'S0')], sample_dfDiff2040.loc[:, ('B0p2Im', 'High', 'S1500')], sample_dfDiff2040.loc[:, ('B0p2Im', 'Med', 'S1500')]], widths=3, positions=[17,19,21,23])
#ax98.violinplot([sample_dfDiff2040.loc[:, ('B1p0Im', 'High', 'S0')], sample_dfDiff2040.loc[:, ('B1p0Im', 'Med', 'S0')], sample_dfDiff2040.loc[:, ('B1p0Im', 'High', 'S1500')], sample_dfDiff2040.loc[:, ('B1p0Im', 'Med', 'S1500')]], widths=3, positions=[97,99,101,103])
#ax98_2.violinplot([sample_dfDiff2040.loc[:, ('B1p0Im', 'High', 'S0')], sample_dfDiff2040.loc[:, ('B1p0Im', 'Med', 'S0')], sample_dfDiff2040.loc[:, ('B1p0Im', 'High', 'S1500')], sample_dfDiff2040.loc[:, ('B1p0Im', 'Med', 'S1500')]], widths=3, positions=[97,99,101,103])
colors = ['#E30B5C', '#C2B280', '#6495ED', 'gold']
i=0
color = colors[1]
# for patch, color in zip(vplot1['bodies'], colors):
for v, color in zip([vplot1,vplot2,vplot3,vplot4],colors):
    for patch in v['bodies']:
        patch.set_color(color)
        # patch.set_facecolor('#D43F3A')
        patch.set_edgecolor(color)
        v['cmaxes'].set_color(color)
        v['cmins'].set_color(color)
        v['cbars'].set_color(color)


#ax98.violinplot([sample_dfDiff2040.loc[:, ('B0p5Im', 'High', 'S0')], sample_dfDiff2040.loc[:, ('B0p5Im', 'Med', 'S0')], sample_dfDiff2040.loc[:, ('B0p5Im', 'High', 'S1500')], sample_dfDiff2040.loc[:, ('B0p5Im', 'Med', 'S1500')]], widths=3, positions=[47,49,51,53])
          # label = 'Low bio, high CS', yerr=abs(errorbars('Med','S1500',mandate,sample_dfDiff2040)-(df9.abs().sum().values - df9['Opt'].sum())))#, elinewidth=errorlinewidth, linewidth = 1.5, color='gold')
# sample_df.xs('FT2', level='biofuel_sensitivity')[(man, biomass,cs)]
# ax98_2.errorbar([0,BtLshare2040.filter(regex='B0p0Im-High.*S0')[0]*100,20,50,100],(df6.abs().sum().values - df6['Opt'].sum()),
#               label = 'High bio, low CS', yerr=abs(errorbars('High','S0',mandate,sample_dfDiff2040)-(df6.abs().sum().values - df6['Opt'].sum())), elinewidth=errorlinewidth, linewidth = 1.5, color='#E30B5C')
# ax98_2.errorbar([0+a,BtLshare2040.filter(regex='B0p0Im-High.*S1500')[0]*100+a,20+a,50+a,100+a],(df7.abs().sum().values - df7['Opt'].sum()),
#           label = 'High bio, high CS', yerr=abs(errorbars('High','S1500',mandate,sample_dfDiff2040)-(df7.abs().sum().values - df7['Opt'].sum())), elinewidth=errorlinewidth, linewidth = 1.5, color='#6495ED')
# ax98_2.errorbar([0-a,BtLshare2040.filter(regex='B0p0Im-Med.*S0')[0]*100-a,20-a,50-a,100-a],(df8.abs().sum().values - df8['Opt'].sum()),
#           label = 'Low bio, low CS', yerr=abs(errorbars('Med','S0',mandate,sample_dfDiff2040)-(df8.abs().sum().values - df8['Opt'].sum())), elinewidth=errorlinewidth, linewidth = 1.5, color='#C2B280')
# ax98_2.errorbar([0-2*a,BtLshare2040.filter(regex='B0p0Im-Med.*S1500')[0]*100-2*a,20-2*a,50-2*a,100-2*a],(df9.abs().sum().values - df9['Opt'].sum()),
#           label = 'Low bio, high CS', yerr=abs(errorbars('Med','S1500',mandate,sample_dfDiff2040)-(df9.abs().sum().values - df9['Opt'].sum())), elinewidth=errorlinewidth, linewidth = 1.5, color='gold')

vplot5 = ax98_2.violinplot(data1, widths=w, positions=[0-3*a,20-3*a,50-3*a,100-3*a])
vplot6 = ax98_2.violinplot(data2, widths=w, positions=[0-a,20-a,50-a,100-a])
vplot7 = ax98_2.violinplot(data3, widths=w, positions=[0+a,20+a,50+a,100+a])
vplot8 = ax98_2.violinplot(data4, widths=w, positions=[0+3*a,20+3*a,50+3*a,100+3*a])

colors = ['#E30B5C', '#C2B280', '#6495ED', 'gold']
i=0
color = colors[1]
# for patch, color in zip(vplot1['bodies'], colors):
for v, color in zip([vplot5,vplot6,vplot7,vplot8],colors):
    for patch in v['bodies']:
        patch.set_color(color)
        # patch.set_facecolor('#D43F3A')
        patch.set_edgecolor(color)
        v['cmaxes'].set_color(color)
        v['cmins'].set_color(color)
        v['cbars'].set_color(color)



# hide the spines between ax and ax2
ax98_2.spines['bottom'].set_visible(False)
ax98.spines['top'].set_visible(False)
#ax98_2.xaxis.tick_top()
ax98_2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax98_2.tick_params(labeltop=False)  # don't put tick labels at the top
ax98.xaxis.tick_bottom()


# sample_df.loc[:, ('High','S400')].plot(kind="errorbar", ax=ax99)
# error = [[min(sample_df.loc[:, ('High','S400')]), max(sample_df.loc[:, ('High','S400')])],
#          [min(sample_df.loc[:, ('High','S400')]), max(sample_df.loc[:, ('High','S400')])],
#          [min(sample_df.loc[:, ('High','S400')]), max(sample_df.loc[:, ('High','S400')])],
#          [min(sample_df.loc[:, ('High','S400')]), max(sample_df.loc[:, ('High','S400')])],
#          [min(sample_df.loc[:, ('High','S400')]), max(sample_df.loc[:, ('High','S400')])]]


# error = [[0,-20, 0, -10, 30],[10,50, 90, 50, 90]]
# ax99.plot(50,50, yerr=[min(sample_df.loc[:, ('High','S400')]), max(sample_df.loc[:, ('High','S400')])], fmt='o')

data5 = [sample_dfDiff2060.loc[:, ('B0p0ImEq', 'High', 'S400')], sample_dfDiff2060.loc[:, ('B0p2Im', 'High', 'S400')], sample_dfDiff2060.loc[:, ('B0p5Im', 'High', 'S400')], sample_dfDiff2060.loc[:, ('B1p0Im', 'High', 'S400')]]
data6 = [sample_dfDiff2060.loc[:, ('B0p0ImEq', 'High', 'S1500')], sample_dfDiff2060.loc[:, ('B0p2Im', 'High', 'S1500')], sample_dfDiff2060.loc[:, ('B0p5Im', 'High', 'S1500')], sample_dfDiff2060.loc[:, ('B1p0Im', 'High', 'S1500')]]
data7 = [sample_dfDiff2060.loc[:, ('B0p0ImEq', 'Med', 'S400')], sample_dfDiff2060.loc[:, ('B0p2Im', 'Med', 'S400')], sample_dfDiff2060.loc[:, ('B0p5Im', 'Med', 'S400')], sample_dfDiff2060.loc[:, ('B1p0Im', 'Med', 'S400')]]
data8 = [sample_dfDiff2060.loc[:, ('B0p0ImEq', 'Med', 'S1500')], sample_dfDiff2060.loc[:, ('B0p2Im', 'Med', 'S1500')], sample_dfDiff2060.loc[:, ('B0p5Im', 'Med', 'S1500')], sample_dfDiff2060.loc[:, ('B1p0Im', 'Med', 'S1500')]]

# print('error: ', errorbars('Med','S400',mandate,sample_dfDiff2060))

#Violin plots
vplot9 = ax99.violinplot(data5, widths=w, positions=[0-3*a,20-3*a,50-3*a,100-3*a])
vplot10 = ax99.violinplot(data6, widths=w, positions=[0-a,20-a,50-a,100-a])
vplot11 = ax99.violinplot(data7, widths=w, positions=[0+a,20+a,50+a,100+a])
vplot12 = ax99.violinplot(data8, widths=w, positions=[0+3*a,20+3*a,50+3*a,100+3*a])

for v, color in zip([vplot9,vplot10,vplot11,vplot12],colors):
    for patch in v['bodies']:
        patch.set_color(color)
        # patch.set_facecolor('#D43F3A')
        patch.set_edgecolor(color)
        v['cmaxes'].set_color(color)
        v['cmins'].set_color(color)
        v['cbars'].set_color(color)

a = 0
ax99.plot([0,BtLshare2060.filter(regex='B0p0Im-High.*S400')[0]*100,20,50,100],(df2.abs().sum().values - df2['Opt'].sum()), label = 'High bio, low CS', color='#E30B5C')
ax99.plot([0+a,BtLshare2060.filter(regex='B0p0Im-High.*S1500')[0]*100+a,20+a,50+a,100+a],(df3.sum().values - df3['Opt'].sum()), label = 'High bio, high CS', color='#6495ED')
ax99.plot([0-a,BtLshare2060.filter(regex='B0p0Im-Med.*S400')[0]*100-a,20-a,50-a,100-a],(df4.sum().values - df4['Opt'].sum()), label = 'Low bio, low CS', color='#C2B280')
ax99.plot([0-2*a,BtLshare2060.filter(regex='B0p0Im-Med.*S1500')[0]*100-2*a,20-2*a,50-2*a,100-2*a],(df5.sum().values - df5['Opt'].sum()), label = 'Low bio, high CS', color='gold')



# ax99.errorbar([0,BtLshare2060.filter(regex='B0p0Im-High.*S400')[0]*100,20,50,100],(df2.abs().sum().values - df2['Opt'].sum()),
#               label = 'High bio, low CS', yerr=abs(errorbars('High','S400',mandate,sample_dfDiff2060) - (df2.abs().sum().values - df2['Opt'].sum())), elinewidth=errorlinewidth, linewidth = 1.5, color='#E30B5C')
# ax99.errorbar([0+a,BtLshare2060.filter(regex='B0p0Im-High.*S1500')[0]*100+a,20+a,50+a,100+a],(df3.sum().values - df3['Opt'].sum()),
#           label = 'High bio, high CS', yerr=abs(errorbars('High','S1500',mandate,sample_dfDiff2060) - (df3.abs().sum().values - df3['Opt'].sum())), elinewidth=errorlinewidth, linewidth = 1.5, color='#6495ED')
# ax99.errorbar([0-a,BtLshare2060.filter(regex='B0p0Im-Med.*S400')[0]*100-a,20-a,50-a,100-a],(df4.sum().values - df4['Opt'].sum()),
#           label = 'Low bio, low CS', yerr=abs(errorbars('Med','S400',mandate,sample_dfDiff2060) - (df4.abs().sum().values - df4['Opt'].sum())), elinewidth=errorlinewidth, linewidth = 1.5, color='#C2B280')
# ax99.errorbar([0-2*a,BtLshare2060.filter(regex='B0p0Im-Med.*S1500')[0]*100-2*a,20-2*a,50-2*a,100-2*a],(df5.sum().values - df5['Opt'].sum()),
#           label = 'Low bio, high CS', yerr=abs(errorbars('Med','S1500',mandate,sample_dfDiff2060) - (df5.abs().sum().values - df5['Opt'].sum())), elinewidth=errorlinewidth, linewidth = 1.5, color='gold')
# ax99.violinplot([sample_dfDiff2060.loc[:, ('B0p5Im', 'High', 'S400')], sample_dfDiff2060.loc[:, ('B0p5Im', 'Med', 'S400')], sample_dfDiff2060.loc[:, ('B0p5Im', 'High', 'S1500')], sample_dfDiff2060.loc[:, ('B0p5Im', 'Med', 'S1500')]], widths=2, positions=[47,49,51,53])
# ax99.violinplot([sample_dfDiff2060.loc[:, ('B1p0Im', 'High', 'S400')], sample_dfDiff2060.loc[:, ('B1p0Im', 'Med', 'S400')], sample_dfDiff2060.loc[:, ('B1p0Im', 'High', 'S1500')], sample_dfDiff2060.loc[:, ('B1p0Im', 'Med', 'S1500')]], widths=2, positions=[97,99,101,103])
#
# ax99_2.errorbar([0,BtLshare2060.filter(regex='B0p0Im-High.*S400')[0]*100,20,50,100],(df2.abs().sum().values - df2['Opt'].sum()),
#               label = 'High bio, low CS', yerr=abs(errorbars('High','S400',mandate,sample_dfDiff2060) - (df2.abs().sum().values - df2['Opt'].sum())), elinewidth=errorlinewidth, linewidth = 1.5, color='#E30B5C')
# ax99_2.errorbar([0+a,BtLshare2060.filter(regex='B0p0Im-High.*S1500')[0]*100+a,20+a,50+a,100+a],(df3.sum().values - df3['Opt'].sum()),
#           label = 'High bio, high CS', yerr=abs(errorbars('High','S1500',mandate,sample_dfDiff2060) - (df3.abs().sum().values - df3['Opt'].sum())), elinewidth=errorlinewidth, linewidth = 1.5, color='#6495ED')
# ax99_2.errorbar([0-a,BtLshare2060.filter(regex='B0p0Im-Med.*S400')[0]*100-a,20-a,50-a,100-a],(df4.sum().values - df4['Opt'].sum()),
#           label = 'Low bio, low CS', yerr=abs(errorbars('Med','S400',mandate,sample_dfDiff2060) - (df4.abs().sum().values - df4['Opt'].sum())), elinewidth=errorlinewidth, linewidth = 1.5, color='#C2B280')
# ax99_2.errorbar([0-2*a,BtLshare2060.filter(regex='B0p0Im-Med.*S1500')[0]*100-2*a,20-2*a,50-2*a,100-2*a],(df5.sum().values - df5['Opt'].sum()),
#           label = 'Low bio, high CS', yerr=abs(errorbars('Med','S1500',mandate,sample_dfDiff2060) - (df5.abs().sum().values - df5['Opt'].sum())), elinewidth=errorlinewidth, linewidth = 1.5, color='gold')


vplot13 = ax99_2.violinplot(data5, widths=w, positions=[0-3*a,20-3*a,50-3*a,100-3*a])
vplot14 = ax99_2.violinplot(data6, widths=w, positions=[0-a,20-a,50-a,100-a])
vplot15 = ax99_2.violinplot(data7, widths=w, positions=[0+a,20+a,50+a,100+a])
vplot16 = ax99_2.violinplot(data8, widths=w, positions=[0+3*a,20+3*a,50+3*a,100+3*a])

for v, color in zip([vplot13,vplot14,vplot15,vplot16],colors):
    for patch in v['bodies']:
        patch.set_color(color)
        # patch.set_facecolor('#D43F3A')
        patch.set_edgecolor(color)
        v['cmaxes'].set_color(color)
        v['cmins'].set_color(color)
        v['cbars'].set_color(color)


# hide the spines between ax and ax2
ax99_2.spines['bottom'].set_visible(False)
ax99.spines['top'].set_visible(False)
# ax99_2.set_xticks([0,500,1000,1500,2000])
# ax99_2.xaxis.tick_top()
ax99_2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax99_2.tick_params(labeltop=False)  # don't put tick labels at the top
ax99.xaxis.tick_bottom()

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax99_2.plot([0, 1], [0, 0], transform=ax99_2.transAxes, **kwargs)
ax99.plot([0, 1], [1, 1], transform=ax99.transAxes, **kwargs)
ax98_2.plot([0, 1], [0, 0], transform=ax98_2.transAxes, **kwargs)
ax98.plot([0, 1], [1, 1], transform=ax98.transAxes, **kwargs)

def mwhpershare2040_forward(x):
    return x * 42.23 #np.interp(0, 37)

def mwhpershare2040_inverse(x):
    return x / 42.23

ax98_2.set_ylim([950, 1120])
ax98.set_ylim([-20, 500])  # snakemake.config['plotting']['costs_max']])
# ax98.set_ylabel('Increase compared to no biofuel mandate [Billion EUR]')  # "System Cost [EUR billion per year]")
fig3.text(0.07,0.5, 'Increase compared to no biofuel mandate [Billion EUR]', va='center', rotation='vertical')  # "System Cost [EUR billion per year]")
ax98.set_xlabel('Biofuel share [%]')
# ax98.set_xscale('symlog')
ax98_2.set_title('2040')
ax98.legend(loc="upper left")#handles, order, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)
secax = ax98_2.secondary_xaxis('top', functions=(mwhpershare2040_forward,mwhpershare2040_inverse))
secax.set_xlabel('Biofuel amount [TWh]')


def mwhpershare2060_forward(x):
    return x * 24.11 #np.interp(0, 37)

def mwhpershare2060_inverse(x):
    return x / 24.11

ax99_2.set_ylim([950, 1100])
ax99.set_ylim([-20, 500])  # snakemake.config['plotting']['costs_max']])
ax99.set_xlabel('Biofuel share [%]')
# ax99.set_xscale('symlog')
ax99_2.set_title('2060')
# ax.grid(axis='x')
secax2 = ax99_2.secondary_xaxis('top', functions=(mwhpershare2060_forward,mwhpershare2060_inverse))
secax2.set_xlabel('Biofuel amount [TWh]')
secax2.set_xticks([0,1000,2000])
fig3.subplots_adjust(hspace=0.05)
# fig3.tight_layout(pad=1)

plt.show()

fig3.savefig('../results/1h_2060_2040_costincreaseAbsolute.pdf', bbox_inches='tight')