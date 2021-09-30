import pandas as pd

from six import iteritems

df = pd.read_csv('../../pypsa-eur/resources/powerplants.csv', index_col=[0])#.drop('id', axis=1)

#Data from https://en.wikipedia.org/wiki/List_of_commercial_nuclear_reactors

print('Correcting Nuclear capacities')
# print(df)
# df.loc[(df.Country == 'SE') & (df.Fueltype == 'Nuclear'), 'DateIn'] = '2001'
# df.loc[(df.Country == 'SE') & (df.Fueltype == 'Nuclear'), 'DateRetrofit'] = '2001'
# df.loc[(df.Country == 'SE') & (df.Fueltype == 'Nuclear') & (df.Name == 'Ringhals'), 'DateIn'] = '2001'


# print(df.loc[(df.Country == 'FR') & (df.Fueltype == 'Nuclear')])#, 'DateIn'])
# print(df.loc[(df.Country == 'FR') & (df.Fueltype == 'Nuclear'), ('Name','DateIn')])

#Nuclear power in France
YearCommercial = {
    'St laurent': 1983,
    'Gravelines': 1985,
    'Paluel': 1986,
    'Penly': 1992,
    'Nogent': 1989,
    'Golfech': 1994,
    'St alban': 1987,
    'Belleville': 1989,
    'Blayais': 1983,
    'Cruas': 1985,
    'Fessenheim': 1978,
    'Flamanville': 2005, #new reactor being built, others 1987
    'Dampierre': 1981,
    'Chinon': 1988,
    'Bugey': 1980,
    'Cattenom': 1992,
    'Chooz': 2000,
    'Civaux': 2002,
    'Tricastin': 1981,
    'Dukovany': 1987,
    'Temelín': 2003,
    'Bohunice': 1985,
    'Mochovce': 2000,
    'Krško': 1983,
    'Ringhals': 1983,
    'Oskarshamn': 1985,
    'Forsmark': 1985,
    'Olkiluoto': 2019}

Capacity = {
    'St laurent': 1830,
    'Gravelines': 5460,
    'Paluel': 5320,
    'Penly': 2660,
    'Nogent': 2620,
    'Golfech': 2620,
    'St alban': 2670,
    'Belleville': 2620,
    'Blayais': 3640,
    'Cruas': 3660,
    'Fessenheim': 0,
    'Flamanville': 4260, #incl. new reactor being built
    'Dampierre': 3560,
    'Chinon': 3620,
    'Bugey': 3580,
    'Cattenom': 5200,
    'Chooz': 3000,
    'Civaux': 2990,
    'Tricastin': 3660,
    'Ringhals': 2166,
    'Oskarshamn': 1400,
    'Forsmark': 3269,
    'Dukovany': 1878,
    'Temelín': 2006,
    'Bohunice': 943,
    'Mochovce': 872,
    'Krško': 688,
    'Olkiluoto': 1600,
    'Brokdorf': 0, #Set German capacitities to zero
    'Emsland': 0,
    'Grohnde': 0,
    'Gundremmingen': 0,
    'Isar': 0,
    'Neckarwestheim': 0}

# fr_nuc = pd.DataFrame(df.loc[(df.Country == "FR") & (df.Fueltype == "Nuclear"),["Name", "DateIn","Capacity"],])
fr_nuc = pd.DataFrame(df.loc[(df.Fueltype == "Nuclear"),["Name", "DateIn","Capacity"],])
for name, year in iteritems(YearCommercial):
    name_match_b = fr_nuc.Name.str.contains(name, case=False, na=False)
    if name_match_b.any():
        fr_nuc.loc[name_match_b, "DateIn"] = year
    else:
        print("'{}' was not found in given DataFrame.".format(name))
    df.loc[fr_nuc.index, "DateIn"] = fr_nuc["DateIn"]

for name, capacity in iteritems(Capacity):
    name_match_b = fr_nuc.Name.str.contains(name, case=False, na=False)
    if name_match_b.any():
        fr_nuc.loc[name_match_b, "Capacity"] = capacity
    else:
        print("'{}' was not found in given DataFrame.".format(name))
    df.loc[fr_nuc.index, "Capacity"] = fr_nuc["Capacity"]

YearRetire = {
        "Grafenrheinfeld": 2015,
        "Philippsburg": 2019,
        "Brokdorf": 2021,
        "Grohnde": 2021,
        "Gundremmingen": 2021,
        "Emsland": 2022,
        "Isar": 2022,
        "Neckarwestheim": 2022,
    }


print(df.loc[(df.Country == 'FR') & (df.Fueltype == 'Nuclear'), ('Name','DateIn','Capacity')])
print(df.loc[(df.Country == 'SE') & (df.Fueltype == 'Nuclear'), 'Capacity'].sum())

# print(df.loc[(df.Country == 'FR') & (df.Fueltype == 'Nuclear'), 'DateIn'])

# print(df.loc[(df.Fueltype == 'Nuclear')].DateIn.head(50))

df.to_csv('../../pypsa-eur/resources/powerplants.csv')