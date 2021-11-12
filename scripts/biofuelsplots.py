import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('ggplot')

sdir = '../results/37h_full_fixedIndustry_noIndustrialHP_2040/csvs/costs.csv'
output = '../results/37h_fixedIndustryNoHP2040.pdf'

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
        'coal': 'fossils',
        'oil': 'fossils',
        'gas': 'fossils',
        'lignite': 'fossils',
        'uranium': 'power',
        'process emissions': 'industry',
        'gas for industry': 'industry',
        'lowT process steam H2': 'industry',
        'SMR': 'other',
        'CC': 'other',
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
        'digestible biomass': 'biomass',
        'solid biomass': 'biomass',
        'biomass to liquid': 'biofuel',
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


def place_subplot(df,ax,ylabel,legend=False):
    new_index = preferred_order2.intersection(df.index).append(df.index.difference(preferred_order2))
    new_columns = df.sum().sort_values().index

    # number_of_techs = 8
    colors = {'power excl. fossils': 'blue',
              'fossils': 'black',
              'other': 'cyan',
              'biomass': 'green',
              'other biomass usage': 'orange',
              'biofuel': 'red',
              'electrofuel': 'darkred',
              'DAC': 'pink'}

    # colormap = plt.cm.jet
    # colors = [colormap(i) for i in np.linspace(0, 1, len(new_index))]
    # colors = sns.color_palette("hls", len(new_index))


    df.loc[new_index,new_columns].T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=colors,#[snakemake.config['plotting']['tech_colors'][i] for i in new_index]
    )

    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_ylim([0,1000])#snakemake.config['plotting']['costs_max']])

    ax.set_ylabel(ylabel)#"System Cost [EUR billion per year]")

    ax.set_xlabel("Biofuel mandate")

    ax.grid(axis='x')

    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1,1], frameon=False)
    if legend==False:
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
    print(df.columns)

cost_df = pd.read_csv(sdir,# skiprows=2,
        index_col=list(range(3)),
        header=list(range(4))
    )

print(cost_df)

df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

df.columns = df.columns.map('|'.join).str.strip('|')
print(df)
#convert to billions
df = df / 1e9

df = df.groupby(df.index.map(rename_techs)).sum()

to_drop = df.index[df.max(axis=1) < 0.5] #snakemake.config['plotting']['costs_threshold']]

print("dropping")

print(df.loc[to_drop])

df = df.drop(to_drop)

print(df)
df2 = df.filter(regex='High.*S400')
df3 = df.filter(regex='High.*S1500')
df4 = df.filter(regex='Med.*S400')
df5 = df.filter(regex='Med.*S1500')

rename_cols(df2)
rename_cols(df3)
rename_cols(df4)
rename_cols(df5)

print(df.sum())
fig, ((ax2, ax3),(ax4,ax5)) = plt.subplots(2,2,figsize=(12,8))

place_subplot(df2,ax2,'High bio, low CCS')
place_subplot(df3,ax3,'High bio, high CCS',legend=True)
place_subplot(df4,ax4,'Med bio, low CCS')
place_subplot(df5,ax5,'Med bio, high CCS')

# axes_handling_left(ax2)
# axes_handling_left(ax4)
# axes_handling_right(ax3,legend=True)
# axes_handling_right(ax5)

fig.savefig(output, bbox_inches='tight')
