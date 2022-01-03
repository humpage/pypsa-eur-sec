import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

scenario = 'serverResults/sensitivities2060_new'
sdir = '../results/{}/csvs/costs.csv'.format(scenario)
output = '../results/sensitivities2060_violin'
balances = '../results/{}/csvs/supply_energy.csv'.format(scenario)
metrics = '../results/{}/csvs/metrics.csv'.format(scenario)


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

            CCUS_DACshare = df.loc['DAC'] / df[df > 0].dropna().sum()
            # print('CCUS DAC share: ', CCUS_DACshare)

            TotalCO2captured = df[df > 0].sum()
            # print('Total CO2 captured: ', TotalCO2captured, ' MtCO2')

        elif k == ['co2']:
            transportCO2emissionShare = (df.loc['electrofuel'] + df.loc['biomass to liquid'] + df.loc['oil emissions'] +
                                         df.loc['shipping oil emissions']) / df[df > 0].dropna().sum()
            # print('Transport share of total CO2 emissions: ', transportCO2emissionShare)

    return H2toEfuelShare, ACtoElectrolysisShare, SolidBiomasstoBtLshare, FossilLiquidFuelsShare, CO2toEfuelShare,\
           CCUS_DACshare, transportCO2emissionShare, TotalCO2captured, BiofuelCO2captureShare, BtLshare

def get_metrics(metrics):

    metrics_df = pd.read_csv(
        metrics,
        index_col=list(range(1)),
        header=list(range(11))
    )

    # metrics = metrics_df / 1e9

    return metrics_df

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

def set_axis_style(ax, labels, ylabels=False):
    ax.yaxis.set_tick_params(direction='out')
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks(np.arange(1, len(labels) + 1))#, labels=labels)
    # if ylabels:
    ax.set_yticklabels(labels)
    # elif not ylabels:
        # ax.set_yticklabels([])
    ax.set_ylim(0.25, len(labels) + 0.75)
#    ax.set_ylabel('Parameter')
    ax.set_xlabel('Total cost increase [%]')
    ax.set_xlim([0, 25])


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

plot_balances(balances)

metrics_df = get_metrics(metrics)


vars = ['FT0', 'FT2', 'E0', 'E2', 'CC0', 'CC2', 'CS0', 'CS2', 'O0', 'O2', 'I0', 'I2']
mandate = ['B0p0', 'B0p2', 'B0p5']
biomass = ['High', 'Med']
carbonstorage = ['S450', 'S1500']

sampledict = {}

for bm, cs in [(bm,cs) for bm in biomass for cs in carbonstorage]:
    sample = metrics_df.xs('37H-T-H-B0p5Im-{}-I-{}-CCL-solar+p3'.format(bm,cs), level='opt', axis=1)
    sampleRef = metrics_df.xs('37H-T-H-B0p0Im-{}-I-{}-CCL-solar+p3'.format(bm,cs), level='opt', axis=1)
    sampledict[bm,cs] = ((sample.loc['total costs'] - sampleRef.loc['total costs']) / sampleRef.loc['total costs']) * 100

sample_df = pd.DataFrame.from_dict(sampledict)

print(sample_df)
sample_test = sample_df.loc[:, ('High','S450')]
sample_test2 = sample_df.loc[:, ('Med','S450')]
# print(sample_test2)


# fig2, ax7 = plt.subplots(nrows=1, ncols=1, figsize=(17, 10))#, sharey=True)
fig2, (ax7, ax8, ax9, ax10) = plt.subplots(nrows=1, ncols=4, figsize=(17, 4), sharey=True)

# print(sample_df.loc[:, ('High','S450')].sort_values())

sample_df.loc[:, ('High','S450')].sort_values().plot(kind="bar", ax=ax7)
sample_df.loc[:, ('Med','S450')].sort_values().plot(kind="bar", ax=ax8)
sample_df.loc[:, ('High','S1500')].sort_values().plot(kind="bar", ax=ax9)
sample_df.loc[:, ('Med','S1500')].sort_values().plot(kind="bar", ax=ax10)
label = ['FT0','FT2','E0','E2','CC0','CC2','CS0','CS2','O0','O2','I0','I2']

for ax in [ax7, ax8, ax9, ax10]:
    set_axis_style_sorted(ax, label)


ax7.set_title('High biomass, low CS')
ax8.set_title('Low biomass, low CS')
ax9.set_title('High biomass, high CS')
ax10.set_title('Low biomass, high CS')
# plt.tick_params(left=False,bottom=False)

fig2.savefig(output + '_sorted.pdf', bbox_inches='tight')
# plt.show()
# d = {}
# for i, j, k, l in [(i, j, k, l) for i in vars for j in biomass for k in carbonstorage for l in mandate]:
#     d[l, "{}_{}".format(j,k), i] = sample.xs('37H-T-H-{}Im-{}-I-{}-CCL-solar+p3'.format(l,j,k,i))#.values


# d = {}
# for i, j, k, l in [(i, j, k, l) for i in vars for j in biomass for k in carbonstorage for l in mandate]:
#     d[l, "{}_{}".format(j,k), i] = sample['37H-T-H-B0p5Im-High-I-S450-CCL-solar+p3'] - sample['37H-T-H-B0p0Im-High-I-S450-CCL-solar+p3']



                                       # .filter(regex='B0p5.*{}.*{}.*{}\|'.format(l,j,k,i)) - sample.filter(regex='B0p0.*{}.*{}.*{}\|'.format(l,j,k,i))
    #.values


# FT0_hiB_loCS = BtLshare.filter(regex='High.*S450.*FT0\|')
# FT2_hiB_loCS = BtLshare.filter(regex='High.*S450.*FT2\|')
# E0_hiB_loCS = BtLshare.filter(regex='High.*S450.*.*E0\|')
# E2_hiB_loCS = BtLshare.filter(regex='High.*S450.*.*E2\|')
# CC0_hiB_loCS = BtLshare.filter(regex='High.*S450.*.*CC0\|')
# CC2_hiB_loCS = BtLshare.filter(regex='High.*S450.*.*CC2\|')
# CS0_hiB_loCS = BtLshare.filter(regex='High.*S450.*.*CS0\|')
# CS2_hiB_loCS = BtLshare.filter(regex='High.*S450.*.*CS2\|')
# O0_hiB_loCS = BtLshare.filter(regex='High.*S450.*.*O0\|')
# O2_hiB_loCS = BtLshare.filter(regex='High.*S450.*.*O2\|')
# I0_hiB_loCS = BtLshare.filter(regex='High.*S450.*I0')
# I2_hiB_loCS = BtLshare.filter(regex='High.*S450.*I2')
# # print(E2_hiB_loCS)
#
# FT0_loB_loCS = BtLshare.filter(regex='Med.*S450.*.*FT0\|')
# FT2_loB_loCS = BtLshare.filter(regex='Med.*S450.*.*FT2\|')
# E0_loB_loCS = BtLshare.filter(regex='Med.*S450.*.*E0\|')
# E2_loB_loCS = BtLshare.filter(regex='Med.*S450.*.*E2\|')
# CC0_loB_loCS = BtLshare.filter(regex='Med.*S450.*.*CC0\|')
# CC2_loB_loCS = BtLshare.filter(regex='Med.*S450.*.*CC2\|')
# CS0_loB_loCS = BtLshare.filter(regex='Med.*S450.*.*CS0\|')
# CS2_loB_loCS = BtLshare.filter(regex='Med.*S450.*.*CS2\|')
# O0_loB_loCS = BtLshare.filter(regex='Med.*S450.*.*O0\|')
# O2_loB_loCS = BtLshare.filter(regex='Med.*S450.*.*O2\|')
# I0_loB_loCS = BtLshare.filter(regex='Med.*S450.*I0')
# I2_loB_loCS = BtLshare.filter(regex='Med.*S450.*I2')
# # print(I0_loB_loCS)
# # print(I2_loB_loCS)
#
# FT0_hiB_hiCS = BtLshare.filter(regex='High.*S1500.*FT0\|')
# FT2_hiB_hiCS = BtLshare.filter(regex='High.*S1500.*FT2\|')
# E0_hiB_hiCS = BtLshare.filter(regex='High.*S1500.*.*E0\|')
# E2_hiB_hiCS = BtLshare.filter(regex='High.*S1500.*.*E2\|')
# CC0_hiB_hiCS = BtLshare.filter(regex='High.*S1500.*.*CC0\|')
# CC2_hiB_hiCS = BtLshare.filter(regex='High.*S1500.*.*CC2\|')
# CS0_hiB_hiCS = BtLshare.filter(regex='High.*S1500.*.*CS0\|')
# CS2_hiB_hiCS = BtLshare.filter(regex='High.*S1500.*.*CS2\|')
# O0_hiB_hiCS = BtLshare.filter(regex='High.*S1500.*.*O0\|')
# O2_hiB_hiCS = BtLshare.filter(regex='High.*S1500.*.*O2\|')
# I0_hiB_hiCS = BtLshare.filter(regex='High.*S1500.*I0')
# I2_hiB_hiCS = BtLshare.filter(regex='High.*S1500.*I2')
# # print(I0_hiB_hiCS)
# # print(I2_hiB_hiCS)
#
# FT0_loB_hiCS = BtLshare.filter(regex='Med.*S1500.*.*FT0\|')
# FT2_loB_hiCS = BtLshare.filter(regex='Med.*S1500.*.*FT2\|')
# E0_loB_hiCS = BtLshare.filter(regex='Med.*S1500.*.*E0\|')
# E2_loB_hiCS = BtLshare.filter(regex='Med.*S1500.*.*E2\|')
# CC0_loB_hiCS = BtLshare.filter(regex='Med.*S1500.*.*CC0\|')
# CC2_loB_hiCS = BtLshare.filter(regex='Med.*S1500.*.*CC2\|')
# CS0_loB_hiCS = BtLshare.filter(regex='Med.*S1500.*.*CS0\|')
# CS2_loB_hiCS = BtLshare.filter(regex='Med.*S1500.*.*CS2\|')
# O0_loB_hiCS = BtLshare.filter(regex='Med.*S1500.*.*O0\|')
# O2_loB_hiCS = BtLshare.filter(regex='Med.*S1500.*.*O2\|')
# I0_loB_hiCS = BtLshare.filter(regex='Med.*S1500.*I0')
# I2_loB_hiCS = BtLshare.filter(regex='Med.*S1500.*I2')
# print(I0_loB_hiCS)
# print(I2_loB_hiCS)

# data1 = [FT0_hiB_loCS, FT2_hiB_loCS, E0_hiB_loCS, E2_hiB_loCS, CC0_hiB_loCS, CC2_hiB_loCS, CS0_hiB_loCS, CS2_hiB_loCS, O0_hiB_loCS, O2_hiB_loCS, I0_hiB_loCS, I2_hiB_loCS]
# data2 = [FT0_loB_loCS, FT2_loB_loCS, E0_loB_loCS, E2_loB_loCS, CC0_loB_loCS, CC2_loB_loCS, CS0_loB_loCS, CS2_loB_loCS, O0_loB_loCS, O2_loB_loCS, I0_loB_loCS, I2_loB_loCS]
# data3 = [FT0_hiB_hiCS, FT2_hiB_hiCS, E0_hiB_hiCS, E2_hiB_hiCS, CC0_hiB_hiCS, CC2_hiB_hiCS, CS0_hiB_hiCS, CS2_hiB_hiCS, O0_hiB_hiCS, O2_hiB_hiCS, I0_hiB_hiCS, I2_hiB_hiCS]
# data4 = [FT0_loB_hiCS, FT2_loB_hiCS, E0_loB_hiCS, E2_loB_hiCS, CC0_loB_hiCS, CC2_loB_hiCS, CS0_loB_hiCS, CS2_loB_hiCS, O0_loB_hiCS, O2_loB_hiCS, I0_loB_hiCS, I2_loB_hiCS]

# print(sample_df)
# print(sample_df.index)
print(sample_df.xs('FT0', level='biofuel_sensitivity')[('High','S450')])
# print(sample_df['FT0'][('High','S450')])


data1 = [sample_df.xs('FT0', level='biofuel_sensitivity')[('High','S450')],
         sample_df.xs('FT2', level='biofuel_sensitivity')[('High', 'S450')],
         sample_df.xs('E0', level='electrolysis_sensitivity')[('High', 'S450')],
         sample_df.xs('E2', level='electrolysis_sensitivity')[('High', 'S450')],
         sample_df.xs('CC0', level='cc_sensitivity')[('High', 'S450')],
         sample_df.xs('CC2', level='cc_sensitivity')[('High', 'S450')],
         sample_df.xs('CS0', level='cs_sensitivity')[('High', 'S450')],
         sample_df.xs('CS2', level='cs_sensitivity')[('High', 'S450')],
         sample_df.xs('O0', level='oil_sensitivity')[('High', 'S450')],
         sample_df.xs('O2', level='oil_sensitivity')[('High', 'S450')],
         sample_df.xs('I0', level='biomass_import_sensitivity')[('High', 'S450')],
         sample_df.xs('I2', level='biomass_import_sensitivity')[('High', 'S450')]]

data2 = [sample_df.xs('FT0', level='biofuel_sensitivity')[('Med','S450')],
         sample_df.xs('FT2', level='biofuel_sensitivity')[('Med', 'S450')],
         sample_df.xs('E0', level='electrolysis_sensitivity')[('Med', 'S450')],
         sample_df.xs('E2', level='electrolysis_sensitivity')[('Med', 'S450')],
         sample_df.xs('CC0', level='cc_sensitivity')[('Med', 'S450')],
         sample_df.xs('CC2', level='cc_sensitivity')[('Med', 'S450')],
         sample_df.xs('CS0', level='cs_sensitivity')[('Med', 'S450')],
         sample_df.xs('CS2', level='cs_sensitivity')[('Med', 'S450')],
         sample_df.xs('O0', level='oil_sensitivity')[('Med', 'S450')],
         sample_df.xs('O2', level='oil_sensitivity')[('Med', 'S450')],
         sample_df.xs('I0', level='biomass_import_sensitivity')[('Med', 'S450')],
         sample_df.xs('I2', level='biomass_import_sensitivity')[('Med', 'S450')]]

data3 = [sample_df.xs('FT0', level='biofuel_sensitivity')[('High','S1500')],
         sample_df.xs('FT2', level='biofuel_sensitivity')[('High', 'S1500')],
         sample_df.xs('E0', level='electrolysis_sensitivity')[('High', 'S1500')],
         sample_df.xs('E2', level='electrolysis_sensitivity')[('High', 'S1500')],
         sample_df.xs('CC0', level='cc_sensitivity')[('High', 'S1500')],
         sample_df.xs('CC2', level='cc_sensitivity')[('High', 'S1500')],
         sample_df.xs('CS0', level='cs_sensitivity')[('High', 'S1500')],
         sample_df.xs('CS2', level='cs_sensitivity')[('High', 'S1500')],
         sample_df.xs('O0', level='oil_sensitivity')[('High', 'S1500')],
         sample_df.xs('O2', level='oil_sensitivity')[('High', 'S1500')],
         sample_df.xs('I0', level='biomass_import_sensitivity')[('High', 'S1500')],
         sample_df.xs('I2', level='biomass_import_sensitivity')[('High', 'S1500')]]

data4 = [sample_df.xs('FT0', level='biofuel_sensitivity')[('Med','S1500')],
         sample_df.xs('FT2', level='biofuel_sensitivity')[('Med', 'S1500')],
         sample_df.xs('E0', level='electrolysis_sensitivity')[('Med', 'S1500')],
         sample_df.xs('E2', level='electrolysis_sensitivity')[('Med', 'S1500')],
         sample_df.xs('CC0', level='cc_sensitivity')[('Med', 'S1500')],
         sample_df.xs('CC2', level='cc_sensitivity')[('Med', 'S1500')],
         sample_df.xs('CS0', level='cs_sensitivity')[('Med', 'S1500')],
         sample_df.xs('CS2', level='cs_sensitivity')[('Med', 'S1500')],
         sample_df.xs('O0', level='oil_sensitivity')[('Med', 'S1500')],
         sample_df.xs('O2', level='oil_sensitivity')[('Med', 'S1500')],
         sample_df.xs('I0', level='biomass_import_sensitivity')[('Med', 'S1500')],
         sample_df.xs('I2', level='biomass_import_sensitivity')[('Med', 'S1500')]]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(17, 4), sharey=True)

# label = ['Fischer-Tropsch low','Fischer-Tropsch high','Electrolysis low','Electrolysis high']
label = ['FT0','FT2','E0','E2','CC0','CC2','CS0','CS2','O0','O2','I0','I2']
for ax in [ax1, ax2, ax3, ax4]:
    set_axis_style(ax, label)


ax1.set_title('High biomass, low CS')
ax2.set_title('Low biomass, low CS')
ax3.set_title('High biomass, high CS')
ax4.set_title('Low biomass, high CS')

bp = ax1.violinplot(data1, vert=0)
bp2 = ax2.violinplot(data2, vert=0)
bp3 = ax3.violinplot(data3, vert=0)
bp4 = ax4.violinplot(data4, vert=0)

plt.show()
fig.savefig(output + '.pdf', bbox_inches='tight')
