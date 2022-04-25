import pandas as pd
import plotly
import chart_studio.plotly as py
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plotly.io.kaleido.scope.mathjax = None

year = '2060'
scenario = 'serverResults/mainScenarios{}'.format(year) #'serverResults/test'
sdir = '../results/{}/csvs/costs.csv'.format(scenario)
output = '../results/1h_{}_fuelSankey'.format(year)
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
    v = [['solid biomass'], ['oil']]#['digestible biomass'],['oil']]#, ['AC'], ['gas'], ['H2']]['oil'],['gas']

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
        # print(df)

        lvl1 = 'lvl1'
        lvl2 = 'lvl2'
        lvl3 = 'lvl3'

        if k in [['solid biomass'], ['digestible biomass'], ['gas']]:
            df = df.loc[~df.index.isin(k)]

        dfTemp = pd.DataFrame(columns=['lvl1', 'lvl2', 'lvl3', 'count', 'color'])

        colors = {'fossil fuel': 'rgba(192, 192, 192, 0.4)',
                  'oil': 'rgba(0, 0, 0, 0.5)',
                  'solid biomass': '#40AA00',
                  'other biomass usage': '#FFE5B4',
                  'biofuel': 'rgba(50,205,50,1)',#'rgba(194,178,128, 1)',
                  'electrofuel': 'rgba(255,215,0, 0.7)',#'pink',#'#832473',
                  'CHP': 'rgba(222,64,50, 0.5)',
                  'BioSNG': 'rgba(255,215,0, 0.2)',
                  'industry heat': 'rgba(144,238,144, 0.2)'#'lightgreen'
        }

        values = df.values
        dfTemp['count'] = values
        dfTemp[lvl1] = df.index
        dfTemp[lvl3] = df.index
        dfTemp[lvl2] = k * len(dfTemp)
        # print(dfTemp['color'])
        # print(mcolors.to_rgba(dfTemp['color'], alpha=0.7))

        for i in range(len(dfTemp)):
            if dfTemp.at[i,'count'] > 0:
                dfTemp.at[i, lvl3] = np.nan
                # dfTemp.at[i, lvl1] = dfTemp.at[i, lvl1] + ' [' + str(int(dfTemp.at[i,'count'])) + ' TWh]'
                # dfTemp.at[i, lvl2] = dfTemp.at[i, lvl2] + ' [' + str(int(dfTemp.at[i,'count'])) + ' TWh]'
                # dfTemp.at[i, lvl3] = dfTemp.at[i, lvl3] + ' [' + str(int(dfTemp.at[i,'count'])) + ' TWh]'
            elif dfTemp.at[i, 'count'] < 0:
                dfTemp.at[i, lvl1] = np.nan
                # dfTemp.at[i, lvl2] = dfTemp.at[i, lvl2] + ' [' + str(int(dfTemp.at[i,'count'])) + ' TWh]'

        dfTemp['color'] = [colors.get(x, '#D3D3D3') for x in df.index]

        print(dfTemp[lvl1])
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

def genSankey(df, domainx, domainy, cat_cols=[], value_cols='', color='', title='Sankey Diagram'):
    # maximum of 6 value cols -> 6 colors

    colorPalette = ['blue', 'blue', 'blue', 'blue', 'blue', 'blue']#'#4B8BBE', '#306998', '#FFE873', '#FFD43B', '#646464', 'gold']
    labelList = []
    colorNumList = []
    for catCol in cat_cols:
        labelListTemp = list(set(df[catCol].values))
        colorNumList.append(len(labelListTemp))
        labelList = labelList + labelListTemp
    print(labelList)

    labelList = list(dict.fromkeys(labelList))

    # define colors based on number of levels
    colorList = []
    for idx, colorNum in enumerate(colorNumList):
        colorList = colorList + [colorPalette[idx]] * colorNum

    # transform df into a source-target pair
    for i in range(len(cat_cols) - 1):
        if i == 0:
            sourceTargetDf = df[[cat_cols[i], cat_cols[i + 1], value_cols, color]]
            sourceTargetDf.columns = ['source', 'target', 'count', 'color']
        else:
            tempDf = df[[cat_cols[i], cat_cols[i + 1], value_cols, color]]
            tempDf.columns = ['source', 'target', 'count', 'color']
            sourceTargetDf = pd.concat([sourceTargetDf, tempDf])
        sourceTargetDf = sourceTargetDf.groupby(['source', 'target', 'color']).agg({'count': 'sum'}).reset_index()

    # add index for source-target pair
    sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(lambda x: labelList.index(x))
    sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(lambda x: labelList.index(x))

    # sourceTargetDf.to_csv('../sourcetarget.csv')
    # creating the sankey diagram
    print(sourceTargetDf)
    data = dict(
        type='sankey',
        node=dict(
            pad=5,
            thickness=1,
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
            value=sourceTargetDf['count'],
            color=sourceTargetDf['color'],
            label=sourceTargetDf['count']
        ),
    domain={
        'x': domainx,
        'y': domainy
    }
    )

    # fig = dict(data=[data], layout=layout)
    return data#, layout#, fig



n_range = 11

if year == '2040':
    df1col = 0
    df2col = 1
    df3col = 12
    df4col = 13
elif year == '2060':
    df1col = 1
    df2col = 2
    df3col = 3
    df4col = 4

df = plot_balances(balances,columnin=df1col)
df2 = plot_balances(balances,columnin=df2col)
df3 = plot_balances(balances,columnin=df3col)
df4 = plot_balances(balances,columnin=df4col)

if year == '2060':
    data = genSankey(df, domainx = [0, 0.48], domainy = [0.52, 1], cat_cols=['lvl1','lvl2','lvl3'],value_cols='count', color='color', title='Energy sankey')
    data2 = genSankey(df2, domainx = [0.52, 1.0], domainy = [0.56, 0.96], cat_cols=['lvl1','lvl2','lvl3'],value_cols='count', color='color', title='Energy sankey')
    data3 = genSankey(df3, domainx = [0, 0.48], domainy = [0.07, 0.42], cat_cols=['lvl1','lvl2','lvl3'],value_cols='count', color='color', title='Energy sankey')
    data4 = genSankey(df4, domainx = [0.52, 1.0], domainy = [0.07, 0.41], cat_cols=['lvl1','lvl2','lvl3'],value_cols='count', color='color', title='Energy sankey')
elif year == '2040':
    data = genSankey(df, domainx = [0, 0.48], domainy = [0.52, 1], cat_cols=['lvl1','lvl2','lvl3'],value_cols='count', color='color', title='Energy sankey')
    data2 = genSankey(df2, domainx = [0.52, 1.0], domainy = [0.52, 1], cat_cols=['lvl1','lvl2','lvl3'],value_cols='count', color='color', title='Energy sankey')
    data3 = genSankey(df3, domainx = [0, 0.48], domainy = [0.02, 0.46], cat_cols=['lvl1','lvl2','lvl3'],value_cols='count', color='color', title='Energy sankey')
    data4 = genSankey(df4, domainx = [0.52, 1.0], domainy = [0.02, 0.46], cat_cols=['lvl1','lvl2','lvl3'],value_cols='count', color='color', title='Energy sankey')

layout = dict(
    # title=title,
    font=dict(
        size=8
    )
)

fig = dict(data=[data, data2, data3, data4], layout=layout)
fig2 = go.Figure(fig)
fig2.add_annotation(x=0.17, y=1,
            text='High biomass, low CS',
            showarrow=False,
            yshift=10,
            align='center')


fig2.add_annotation(x=0.84, y=1,
            text='High biomass, high CS',
            showarrow=False,
            yshift=10,
            align='center')


fig2.add_annotation(x=0.17, y=0.45,
            text='Low biomass, low CS',
            showarrow=False,
            yshift=10,
            align='center')


fig2.add_annotation(x=0.84, y=0.45,
            text='Low biomass, high CS',
            showarrow=False,
            yshift=10,
            align='center')
fig2.show()

# fig = make_subplots(
#     rows=2, cols=2, subplot_titles=('High biomass, low CS', 'High biomass, high CS', 'Low biomass, low CS','Low biomass, high CS'),
#     specs=[[{"type": "domain"}, {"type": "domain"}],
#            [{"type": "domain"}, {"type": "domain"}]],
#     shared_xaxes=True,
#     shared_yaxes=True
# )

# fig.add_trace(go.Sankey(data),
#               row=1, col=1)
# fig.add_trace(go.Sankey(data2),
#               row=1, col=2)
# fig.add_trace(go.Sankey(data3),
#               row=2, col=1)
# fig.add_trace(go.Sankey(data4),
#               row=2, col=2)

fig2.write_image(output + '.pdf')
# plotly.offline.plot(fig, validate=False)