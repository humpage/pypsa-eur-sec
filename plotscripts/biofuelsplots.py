import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('ggplot')

scenario = 'serverResults/mainScenarios'
sdir2060 = '../results/serverResults/mainScenarios2060_update/csvs/costs.csv'
sdir2040 = '../results/serverResults/mainScenarios2040_update/csvs/costs.csv'
output = '../results/1h_2060'
metrics2040 = '../results/{}2040_update/csvs/metrics.csv'.format(scenario)
metrics2060 = '../results/{}2060_update/csvs/metrics.csv'.format(scenario)
balances2040 = '../results/{}2040_update/csvs/supply_energy.csv'.format(scenario)
balances2060 = '../results/{}2060_update/csvs/supply_energy.csv'.format(scenario)


def plot_balances(balances):
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
        print(df)
        # remove trailing link ports
        df.index = [i[:-1] if ((i != "co2") and (i[-1:] in ["0", "1", "2", "3", "4"])) else i for i in df.index]
        print(df)
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


#consolidate and rename
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
        'heat': 'heat and industry',
        'industry': 'heat and industry',
        'CHP': 'heat and industry',
        'power': 'power excl. fossil fuels',
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
    'heat and industry',
    'other',
    'fossil fuels',
    'H2 usages',
    'biomass domestic',
    'biomass import',
    'biomass to liquid',
    'other biomass usage',
])


def axes_handling_left(ax):
    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    # ax.set_ylim([0,snakemake.config['plotting']['costs_max']])

    ax.set_ylabel("System Cost [EUR billion per year]")

    ax.set_xlabel("")

    ax.grid(axis='x')

    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1,1], frameon=False)
    ax.get_legend().remove()


def axes_handling_right(ax,legend=False):
    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    # ax.set_ylim([0,snakemake.config['plotting']['costs_max']])

    ax.set_xlabel("")
    ax.grid(axis='x')

    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1,1], frameon=False)

    if legend==False:
        ax.get_legend().remove()


def place_subplot(df,ax,ylabel,xlabel,ymax,title,legend=False):
    new_index = preferred_order2.intersection(df.index).append(df.index.difference(preferred_order2))
    new_columns = df.sum().index#.sort_values().index

    # number_of_techs = 8
    # colors = {'power excl. fossils': 'blue',
    #           'fossils': 'black',
    #           'other': 'cyan',
    #           'biomass': 'green',
    #           'other biomass usage': 'orange',
    #           'biofuel': 'red',
    #           'electrofuel': 'darkred',
    #           'DAC': 'pink'}

    colors = {'power excl. fossil fuels': '#0057B7',#'#235ebc', #'#6495ED'
              'fossil fuels': '#C0C0C0',
              'fossil liquid fuel': 'grey',
              'fossil fuel + CCS': 'grey',
              'fossil liquid fuel + CCS': '#708090',
              'fossil liquid fuel w/o CCS': '#708090',
              'fossil gas': 'darkgrey',
              'heat and industry': '#FFD700',#'#FAFA33',#'#CD7F32',#'lightblue',
              'other': '#FAFA33',#'#CD7F32',#'lightblue',
              'biomass': 'green',
              'other biomass usage': '#FFE5B4',
              'biofuel': 'gold',
              'electrofuel': '#C2B280',
              'electrofuel + CC': 'lightgreen',
              'DAC': 'pink',
              'carbon storage': 'darkgreen',
              'opportunity cost total': 'blue',
              'opportunity cost': '#C04000',
              'fuel delta': 'green',
              'biomass domestic': '#40AA00',
              'biomass import': '#48CC22',
              'biofuel process': '#ADFF2F',#'lightgreen',
              'electrofuel process': '#E30B5C',  # 'lightgreen',
              # 'electrofuel + CC': '#832473',  # 'lightgreen',
              # 'carbon storage': 'darkgreen'
              }

    # colors = {'power excl. fossils': '#235ebc',
    #           'fossil fuels': 'grey',
    #           'fossil liquid fuel': 'grey',
    #           'fossil fuel + CCS': 'grey',
    #           'fossil gas': 'darkgrey',
    #           'other': 'lightblue',
    #           'biomass domestic': '#40AA00',
    #           'biomass import': '#48CC22',
    #           'other biomass usage': 'red',
    #           'biofuel process': 'orange',
    #           'electrofuel process': '#832473',#'lightgreen',
    #           'electrofuel + CC': '#832473',#'lightgreen',
    #           'DAC': 'pink',
    #           'carbon storage': 'darkgreen'}

    # colormap = plt.cm.jet
    # colors = [colormap(i) for i in np.linspace(0, 1, len(new_index))]
    # colors = sns.color_palette("hls", len(new_index))

    to_plot = df.loc[new_index,new_columns].T
    to_plot.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=colors,#[snakemake.config['plotting']['tech_colors'][i] for i in new_index]
    )
    # for p in ax.patches:
    #     ax.annotate(str(int(p.get_height().sum())), (p.get_x() * 1.005, p.get_height() * 1.005))

    list_values = (to_plot.astype(int).values.flatten('F'))
    for rect, value in zip(ax.patches, list_values):
        if value >= 6:
            h = rect.get_height() / 2.
            w = rect.get_width() / 2.
            x, y = rect.get_xy()
            # ax.text(x + w, y + h, value, horizontalalignment='center', verticalalignment='center', fontsize='xx-small')

    totals = to_plot.sum(axis=1)
    #print(totals.values)

    i=0
    for rect, total in zip(ax.patches, totals):
        #print(total)
        # if i == 0:
        ax.text(rect.get_x()+rect.get_width()/2, total+20, int(total), ha='center', weight='bold')
        # if i > 0:
        #     increase = int(round((total / totals[0]-1) * 100))
        #     print(increase)
        #     ax.text(rect.get_x()+rect.get_width()/2, total+70, '+{}%'.format(increase), ha='center', fontsize='x-small')
        i+=1

    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_ylim([0,ymax])#snakemake.config['plotting']['costs_max']])

    ax.set_ylabel(ylabel)#"System Cost [EUR billion per year]")

    ax.set_xlabel(xlabel)

    ax.set_title(title)
    ax.grid(axis='x')

    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1,1], frameon=False)
    if legend==False:
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
    return df

    # df.columns = df.columns.str.replace('.*-H-.*'.format(str(num)), 'NoBio')
    #print(df.columns)

def rename_cols_onlyOptimal(df):
    df.columns = df.columns.str.replace('.*High.*S400.*', 'High bio, low CS')
    df.columns = df.columns.str.replace('.*High.*S1500.*', 'High bio, high CS')
    df.columns = df.columns.str.replace('.*Med.*S400.*', 'Low bio, low CS')
    df.columns = df.columns.str.replace('.*Med.*S1500.*', 'Low bio, high CS')

    df.columns = df.columns.str.replace('.*High.*S0.*', 'High bio, low CS')
    df.columns = df.columns.str.replace('.*Med.*S0.*', 'Low bio, low CS')

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

def reorder_df(df,order):
    df = df[order]
    return df

def drop_cols(df):
    to_drop = df.filter(regex='S420')
    df.drop(to_drop)
    return df


n_range = 11

df2060 = cost_dataframe(sdir2060)
df2040 = cost_dataframe(sdir2040)

# df.to_csv('../results/costoutput.csv')

# to_drop = df.index[df.max(axis=1) < 0.5] #snakemake.config['plotting']['costs_threshold']]

# print("dropping")

# print(df.loc[to_drop])

# df = df.drop(to_drop)

BtLshare2060 = plot_balances(balances2060)
# print(BtLshare2060.filter(regex='B0p0Im-'))
print(BtLshare2060.filter(regex='B0p0Im-High.*S400')[0])#.values)
BtLshare2040 = plot_balances(balances2040)

# print(df2060)
df2 = df2060.filter(regex='High.*S400')
df3 = df2060.filter(regex='High.*S1500')
df4 = df2060.filter(regex='Med.*S400')
df5 = df2060.filter(regex='Med.*S1500')

df6 = df2040.filter(regex='High.*S0')
df7 = df2040.filter(regex='High.*S1500')
df8 = df2040.filter(regex='Med.*S0')
df9 = df2040.filter(regex='Med.*S1500')
# df2.to_csv('../datadf2.csv')

order=['0%','Opt','20%','50%','100%']
df2 = rename_cols(df2, order)
df3 = rename_cols(df3, order)
df4 = rename_cols(df4, order)
df5 = rename_cols(df5, order)

df6 = rename_cols(df6, order)
df7 = rename_cols(df7, order)
df8 = rename_cols(df8, order)
df9 = rename_cols(df9, order)

print(df2060.sum())
fig, ((ax2, ax3),(ax4,ax5)) = plt.subplots(2,2,figsize=(12,8))

place_subplot(df2,ax2,'Energy system cost [Billion EUR]','',750,'High bio, low CS')
place_subplot(df3,ax3,'','',750,'High bio, high CS',legend=True)
place_subplot(df4,ax4,'Energy system cost [Billion EUR]','Biofuel mandate',750,'Low bio, low CS')
place_subplot(df5,ax5,'','Biofuel mandate',750,'Low bio, high CS')

fig.tight_layout(pad=1)

fig.savefig(output + '.pdf', bbox_inches='tight')


fig4, ((ax6, ax7),(ax8,ax9)) = plt.subplots(2,2,figsize=(12,8))

# place_subplot(df6,ax6,'High bio, low CS','',1050,'(a)')
# place_subplot(df7,ax7,'High bio, high CS','',1050,'(b)',legend=True)
# place_subplot(df8,ax8,'Low bio, low CS','Biofuel mandate',1050,'(c)')
# place_subplot(df9,ax9,'Low bio, high CS','Biofuel mandate',1050,'(d)')


place_subplot(df6,ax6,'Energy system cost [Billion EUR]','',1050,'High bio, low CS')
place_subplot(df7,ax7,'','',1050,'High bio, high CS',legend=True)
place_subplot(df8,ax8,'Energy system cost [Billion EUR]','Biofuel mandate',1050,'Low bio, low CS')
place_subplot(df9,ax9,'','Biofuel mandate',1050,'Low bio, high CS')

fig4.tight_layout(pad=1)

fig4.savefig('../results/1h_2040.pdf', bbox_inches='tight')


fig3, (ax98,ax99) = plt.subplots(1,2,figsize=(12,5))

# ax98.plot([0,50,100],100 * (df6.sum().values - min(df6.sum().values)) / min(df6.sum().values), label = 'High bio, low CS')
# ax98.plot([0,50,100],100 * (df7.sum().values - min(df7.sum().values)) / min(df7.sum().values), label = 'High bio, high CS')
# ax98.plot([0,50,100],100 * (df8.sum().values - min(df8.sum().values)) / min(df8.sum().values), label = 'Low bio, low CS')
# ax98.plot([0,50,100],100 * (df9.sum().values - min(df9.sum().values)) / min(df9.sum().values), label = 'Low bio, high CS')
#
# ax99.plot([0,20,50,100],100 * (df2.sum().values - min(df2.sum().values)) / min(df2.sum().values), label = 'High bio, low CS')
# ax99.plot([0,20,50,100],100 * (df3.sum().values - min(df3.sum().values)) / min(df3.sum().values), label = 'High bio, high CS')
# ax99.plot([0,50,100],100 * (df4.sum().values - min(df4.sum().values)) / min(df4.sum().values), label = 'Low bio, low CS')
# ax99.plot([0,50,100],100 * (df5.sum().values - min(df5.sum().values)) / min(df5.sum().values), label = 'Low bio, high CS')

print('DF4: ',df4.sum())
print(df4['Opt'].sum())
ax98.plot([0,BtLshare2040.filter(regex='B0p0Im-High.*S0')[0]*100,20,50,100],(df6.sum().values - df6['Opt'].sum()), label = 'High bio, low CS')
ax98.plot([0,BtLshare2040.filter(regex='B0p0Im-High.*S1500')[0]*100,20,50,100],(df7.sum().values - df7['Opt'].sum()), label = 'High bio, high CS')
ax98.plot([0,BtLshare2040.filter(regex='B0p0Im-Med.*S0')[0]*100,20,50,100],(df8.sum().values - df8['Opt'].sum()), label = 'Low bio, low CS')
ax98.plot([0,BtLshare2040.filter(regex='B0p0Im-Med.*S1500')[0]*100,20,50,100],(df9.sum().values - df9['Opt'].sum()), label = 'Low bio, high CS')

ax99.plot([0,BtLshare2060.filter(regex='B0p0Im-High.*S400')[0]*100,20,50,100],(df2.sum().values - df2['Opt'].sum()), label = 'High bio, low CS')
ax99.plot([0,BtLshare2060.filter(regex='B0p0Im-High.*S1500')[0]*100,20,50,100],(df3.sum().values - df3['Opt'].sum()), label = 'High bio, high CS')
ax99.plot([0,BtLshare2060.filter(regex='B0p0Im-Med.*S400')[0]*100,20,50,100],(df4.sum().values - df4['Opt'].sum()), label = 'Low bio, low CS')
ax99.plot([0,BtLshare2060.filter(regex='B0p0Im-Med.*S1500')[0]*100,20,50,100],(df5.sum().values - df5['Opt'].sum()), label = 'Low bio, high CS')


def mwhpershare2040_forward(x):
    return x * 42.23 #np.interp(0, 37)

def mwhpershare2040_inverse(x):
    return x / 42.23

ax98.set_ylim([-20, 500])  # snakemake.config['plotting']['costs_max']])
ax98.set_ylabel('Increase compared to no biofuel mandate [Billion EUR]')  # "System Cost [EUR billion per year]")
ax98.set_xlabel('Biofuel share [%]')
# ax98.set_xscale('symlog')
ax98.set_title('2040')
secax = ax98.secondary_xaxis('top', functions=(mwhpershare2040_forward,mwhpershare2040_inverse))
secax.set_xlabel('Biofuel amount [TWh]')


def mwhpershare2060_forward(x):
    return x * 20.11 #np.interp(0, 37)

def mwhpershare2060_inverse(x):
    return x / 20.11

ax99.set_ylim([-20, 500])  # snakemake.config['plotting']['costs_max']])
ax99.set_xlabel('Biofuel share [%]')
# ax99.set_xscale('symlog')
ax99.set_title('2060')
# ax.grid(axis='x')
ax99.legend(loc="upper left")#handles, order, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)
secax2 = ax99.secondary_xaxis('top', functions=(mwhpershare2060_forward,mwhpershare2060_inverse))
secax2.set_xlabel('Biofuel amount [TWh]')
# plt.show()

fig3.savefig(output + '_2040_costincreaseAbsolute.pdf', bbox_inches='tight')

metrics2040_df = get_metrics(metrics2040)
metrics2060_df = get_metrics(metrics2060)
#print(metrics2060_df.loc['co2_shadow'])


co2_df1 = metrics2040_df.loc['co2_shadow'].filter(regex='High.*S0')
co2_df2 = metrics2040_df.loc['co2_shadow'].filter(regex='High.*S1500')
co2_df3 = metrics2040_df.loc['co2_shadow'].filter(regex='Med.*S0')
co2_df4 = metrics2040_df.loc['co2_shadow'].filter(regex='Med.*S1500')

co2_df5 = metrics2060_df.loc['co2_shadow'].filter(regex='High.*S400')
co2_df6 = metrics2060_df.loc['co2_shadow'].filter(regex='High.*S1500')
co2_df7 = metrics2060_df.loc['co2_shadow'].filter(regex='Med.*S400')
co2_df8 = metrics2060_df.loc['co2_shadow'].filter(regex='Med.*S1500')

print(co2_df1)
print(co2_df7)
# co2_df1 = rename_cols(co2_df1, order)
# co2_df2 = rename_cols(co2_df2, order)
# co2_df3 = rename_cols(co2_df3, order)
# co2_df4 = rename_cols(co2_df4, order)
# co2_df5 = rename_cols(co2_df5, order)
# co2_df6 = rename_cols(co2_df6, order)
# co2_df7 = rename_cols(co2_df7, order)
# co2_df8 = rename_cols(co2_df8, order)

#print(co2_df1.values)

# order = ['High bio, low CS','High bio, high CS','Low bio, low CS','Low bio, high CS']
fig4, (ax198, ax199) = plt.subplots(1,2,figsize=(12,5))
ax198.plot([0,BtLshare2040.filter(regex='B0p0Im-High.*S0')[0]*100,20,50,100],co2_df1.values, label = 'High bio, low CS')
ax198.plot([0,BtLshare2040.filter(regex='B0p0Im-High.*S1500')[0]*100,20,50,100],co2_df2.values, label = 'High bio, high CS')
ax198.plot([0,BtLshare2040.filter(regex='B0p0Im-Med.*S0')[0]*100,20,50,100],co2_df3.values, label = 'Low bio, low CS')
ax198.plot([0,BtLshare2040.filter(regex='B0p0Im-Med.*S1500')[0]*100,20,50,100],co2_df4.values, label = 'Low bio, high CS')

ax199.plot([0,BtLshare2060.filter(regex='B0p0Im-High.*S400')[0]*100,20,50,100],co2_df5.values, label = 'High bio, low CS')
ax199.plot([0,BtLshare2060.filter(regex='B0p0Im-High.*S1500')[0]*100,20,50,100],co2_df6.values, label = 'High bio, high CS')
ax199.plot([0,BtLshare2060.filter(regex='B0p0Im-Med.*S400')[0]*100,20,50,100],co2_df7.values, label = 'Low bio, low CS')
ax199.plot([0,BtLshare2060.filter(regex='B0p0Im-Med.*S1500')[0]*100,20,50,100],co2_df8.values, label = 'Low bio, high CS')

ax198.set_ylim([0, 200])  # snakemake.config['plotting']['costs_max']])
ax198.set_ylabel('CO2 shadow price')  # "System Cost [EUR billion per year]")
ax198.set_xlabel('Biofuel mandate [%]')
ax198.set_title('2040')

ax199.set_ylim([0, 200])  # snakemake.config['plotting']['costs_max']])
ax199.set_ylabel('')  # "System Cost [EUR billion per year]")
ax199.set_xlabel('Biofuel mandate [%]')
ax199.set_title('2060')
ax199.legend()

fig4.savefig(output + '_2040_co2price.pdf', bbox_inches='tight')

# axes_handling_left(ax2)
# axes_handling_left(ax4)
# axes_handling_right(ax3,legend=True)
# axes_handling_right(ax5)
fig.tight_layout(pad=1.5)
plt.show()
fig.savefig(output + '.pdf', bbox_inches='tight')

df2040 = cost_dataframe(sdir2040)

dfOpt2060 = df2060.filter(regex='B0p0Im-')
dfOpt2060 = dfOpt2060.filter(regex='^(?!.*S420).*$')
dfOpt2040 = df2040.filter(regex='B0p0Im-')
# dfOpt2060 = drop_cols(dfOpt2060)

rename_cols_onlyOptimal(dfOpt2060)
rename_cols_onlyOptimal(dfOpt2040)
# order=['High bio, high CS', 'High bio, low CS', 'Low bio, high CS', 'Low bio, low CS']
# dfOpt2060 = reorder_df(dfOpt2060,order)
# dfOpt2040 = reorder_df(dfOpt2040,order)

fig2, (ax6, ax7) = plt.subplots(1,2,figsize=(12,5))
place_subplot(dfOpt2060,ax7,'','',650,'2060', legend=True)
place_subplot(dfOpt2040,ax6,'Total cost (billion EUR)','',650,'2040')

fig2.tight_layout(pad=1.5)
plt.show()

fig2.savefig(output + '_2040_optimum.pdf', bbox_inches='tight')