import pandas as pd
import plotly
import chart_studio.plotly as py
import numpy as np
import plotly.graph_objs as go

import matplotlib.pyplot as plt
plotly.io.kaleido.scope.mathjax = None


scenario = 'serverResults/mainScenarios2060' #'serverResults/test'
sdir = '../results/{}/csvs/costs.csv'.format(scenario)
output = '../results/1h_2060_allSankeyTest'
balances = '../results/{}/csvs/supply_energy.csv'.format(scenario)


def plot_balances(balances,columnin):
    co2_carriers = ["co2", "co2 stored", "process emissions"]

    balances_df = pd.read_csv(
        balances,
        index_col=list(range(3)),
        header=list(range(n_range))
    )

    balances_df.columns = balances_df.columns.map('|'.join).str.strip('|')
    balances = {i.replace(" ", "_"): [i] for i in balances_df.index.levels[0]}
    balances["energy"] = [i for i in balances_df.index.levels[0] if i not in co2_carriers]

    dfTot = pd.DataFrame(columns=['lvl1', 'lvl2', 'lvl3', 'count'])
    # v = [['solid biomass'], ['digestible biomass']]#, ['AC'], ['gas'], ['H2']]['oil'],['gas']
    v = [['solid biomass'], ['digestible biomass'],['oil']]#, ['AC'], ['gas'], ['H2']]['oil'],['gas']

    # v = balances_df.index.levels[0].drop(co2_carriers)
    # v = v.drop(['uranium'])

    for k in v:
        df = balances_df.loc[k]
        df = df.groupby(df.index.get_level_values(2)).sum()

        # convert MWh to TWh
        df = df / 1e6

        # remove trailing link ports
        df.index = [i[:-1] if ((i != "co2") and (i[-1:] in ["0", "1", "2", "3", "4"])) else i for i in df.index]
        df = df.groupby(df.index.map(rename_techs_balances)).sum()
        df = df[df.columns[columnin]]

        lvl1 = 'lvl1'
        lvl2 = 'lvl2'
        lvl3 = 'lvl3'

        if k in [['solid biomass'], ['digestible biomass'], ['gas']]:
            df = df.loc[~df.index.isin(k)]

        dfTemp = pd.DataFrame(columns=['lvl1', 'lvl2', 'lvl3', 'count'])

        values = df.values
        dfTemp['count'] = values
        dfTemp[lvl1] = df.index
        dfTemp[lvl3] = df.index
        dfTemp[lvl2] = k * len(dfTemp)

        for i in range(len(dfTemp)):
            if dfTemp.at[i,'count'] > 0:
                dfTemp.at[i, lvl3] = np.nan
            elif dfTemp.at[i, 'count'] < 0:
                dfTemp.at[i, lvl1] = np.nan


        # Remove flows less than limit
        limit2 = 5 #TWh
        dfTemp['count'] = abs(dfTemp['count'])
        dfTemp[dfTemp['count'] < limit2] = np.nan

        #If want to show all flows and set them to one
        # dfTemp['count'] = abs(dfTemp['count'])
        # dfTemp['count'] = 1

        dfTot = dfTot.append(dfTemp, ignore_index=True)
        #dfTot.to_csv('../sankeydata.csv')

    return dfTot


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
        "CC": "CC",
        'industry wood': 'domestic biomass',
        'forest residues': 'domestic biomass',
    }

    rename = {
        "solar": "solar PV",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        # "offwind-ac": "offshore wind (AC)",
        # "offwind-dc": "offshore wind (DC)",
        "offwind-ac": "offshore wind",
        "offwind-dc": "offshore wind",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "co2 Store": "DAC",
        "co2 stored": "CO2 sequestration",
        "AC": "transmission lines",
        "DC": "transmission lines",
        "B2B": "transmission lines",
        'oil': 'fossil fuel',
        # 'solid biomass': 'solid biomass residues',
        'solid biomass for mediumT industry': 'industry heat',
        'lowT process steam solid biomass': 'industry heat',
        'lowT process steam electricity': 'industry heat',
        'ground heat pump': 'heating',
        'air heat pump': 'heating',
        'resistive heater': 'heating',
        'gas for highT industry': 'industry heat',
        'gas for mediumT industry': 'industry heat',
        'lowT process steam methane': 'industry heat',
        'gas boiler': 'heating',
        'forest residues solid biomass': 'domestic solid biomass',
        'industry wood residues solid biomass': 'domestic solid biomass',
        'landscape care solid biomass': 'domestic solid biomass',
        'highT industry': 'process heat',
        'mediumT industry': 'process heat',
        'H2 for industry': 'industry feedstock',
        'naphtha for industry': 'naphtha',
        'H2 for shipping': 'shipping',
        'shipping oil': 'shipping',
        'kerosene for aviation': 'aviation',
        'biomass to liquid': 'biofuel',
        'land transport oil': 'land transport',
        'land transport fuel cell': 'land transport',
        'BEV charger': 'land transport',
        'H2 liquefaction': 'shipping',
        'nuclear_new': 'nuclear'
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

def genSankey(df, domainx, domainy, cat_cols=[], value_cols='', title='Sankey Diagram'):
    # maximum of 6 value cols -> 6 colors
    colorPalette = ['blue', 'blue', 'blue', 'blue', 'blue', 'blue']#'#4B8BBE', '#306998', '#FFE873', '#FFD43B', '#646464', 'gold']
    labelList = []
    colorNumList = []
    for catCol in cat_cols:
        labelListTemp = list(set(df[catCol].values))
        colorNumList.append(len(labelListTemp))
        labelList = labelList + labelListTemp

    # remove duplicates from labelList
    labelList = list(dict.fromkeys(labelList))

    # define colors based on number of levels
    colorList = []
    for idx, colorNum in enumerate(colorNumList):
        colorList = colorList + [colorPalette[idx]] * colorNum

    # transform df into a source-target pair
    for i in range(len(cat_cols) - 1):
        if i == 0:
            sourceTargetDf = df[[cat_cols[i], cat_cols[i + 1], value_cols]]
            sourceTargetDf.columns = ['source', 'target', 'count']
        else:
            tempDf = df[[cat_cols[i], cat_cols[i + 1], value_cols]]
            tempDf.columns = ['source', 'target', 'count']
            sourceTargetDf = pd.concat([sourceTargetDf, tempDf])
        sourceTargetDf = sourceTargetDf.groupby(['source', 'target']).agg({'count': 'sum'}).reset_index()

    # add index for source-target pair
    sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(lambda x: labelList.index(x))
    sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(lambda x: labelList.index(x))
    # sourceTargetDf.to_csv('../sourcetarget.csv')
    # creating the sankey diagram
    data = dict(
        type='sankey',
        node=dict(
            pad=5,
            thickness=5,
            line=dict(
                color="black",
                width=0
            ),
            label=labelList,
            color=colorList
        ),
        link=dict(
            source=sourceTargetDf['sourceID'],
            target=sourceTargetDf['targetID'],
            value=sourceTargetDf['count']
        ),
    domain={
        'x': domainx,
        'y': domainy
    }
    )

    # fig = dict(data=[data], layout=layout)
    return data#, layout#, fig



n_range = 11

df = plot_balances(balances,columnin=0)
df2 = plot_balances(balances,columnin=1)
df3 = plot_balances(balances,columnin=8)
df4 = plot_balances(balances,columnin=9)

data = genSankey(df, domainx = [0, 0.45], domainy = [0.55, 1], cat_cols=['lvl1','lvl2','lvl3'],value_cols='count',title='Energy sankey')
data2 = genSankey(df2, domainx = [0.55, 1.0], domainy = [0.55, 1], cat_cols=['lvl1','lvl2','lvl3'],value_cols='count',title='Energy sankey')
data3 = genSankey(df3, domainx = [0, 0.45], domainy = [0, 0.45], cat_cols=['lvl1','lvl2','lvl3'],value_cols='count',title='Energy sankey')
data4 = genSankey(df4, domainx = [0.55, 1.0], domainy = [0, 0.45], cat_cols=['lvl1','lvl2','lvl3'],value_cols='count',title='Energy sankey')

layout = dict(
    # title=title,
    font=dict(
        size=8
    )
)

fig = dict(data=[data, data2, data3, data4], layout=layout)
fig2 = go.Figure(fig, shared_xaxes='all')

fig2.write_image(output + '.pdf')
plotly.offline.plot(fig, validate=False)