import json, urllib
#import chart_studio.plotly as py
import pandas as pd
import numpy as np
from plotly.offline import iplot

refugee_df = pd.read_csv('../../sankeydata.csv')

data_trace = dict(
    type='sankey',
    domain = dict(
      x =  [0,1],
      y =  [0,1]
    ),
    orientation = "h",
    valueformat = ".0f",
    node = dict(
      pad = 10,
      thickness = 30,
      line = dict(
        color = "black",
        width = 0.5
      ),
      label =  refugee_df['Node, Label'].dropna(axis=0, how='any'),
      color = refugee_df['Color']
    ),
    link = dict(
      source = refugee_df['Source'].dropna(axis=0, how='any'),
      target = refugee_df['Target'].dropna(axis=0, how='any'),
      value = refugee_df['Value'].dropna(axis=0, how='any'),
  )
)

layout =  dict(
    title = "Refugee movement through Manus and Nauru, via <a href='http://www.bryanbrussee.com/sankey.html'>Bryan Brussee</a>",
    height = 772,
    width = 950,
    font = dict(
      size = 10
    ),
)


# fig = dict(data=[data_trace], layout=layout)
# py.iplot(fig, validate=False)

data_trace['orientation'] = 'v'
py.iplot([data_trace], layout=layout)
