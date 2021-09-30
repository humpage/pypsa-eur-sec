import pandas as pd

df = pd.read_csv('../../pypsa-eur/resources/powerplants.csv', index_col=[0])#.drop('id', axis=1)


print(df.loc[(df.Country == 'DE') & (df.Fueltype == 'Nuclear'), ('Name','DateIn','Capacity')])
