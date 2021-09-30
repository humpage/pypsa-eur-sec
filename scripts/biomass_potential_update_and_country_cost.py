import pandas as pd
import numpy as np

base_dir="../data/biomass"

#Scenarios: ENS_Low, ENS_Ref, ENS_High, ENS_BaU_GFTM
#Years: 2010, 2020, ... , 2050

def update_biomass_potentials():

    renameColsForest = {'Total supply (PJ)':'forest residues',
                    '>300 km ':'400',
                    'Transport variable cost: Paved road (Euros/GJ* km)': 'TransC',
                    'ForestC (fixed cost, cost at the road and truck loading/uploading) (Euros/GJ)': 'FixC'}
    renameColsAgric = {'Total supply (PJ)':'straw',
                    '>300 km ':'400',
                   'Transport variable cost: Paved road  (Euros/GJ* km)': 'TransC',
                   'AgriC (fixed cost, cost at the road and truck loading/uploading) (Euros/GJ)': 'FixC'}

    forest_residues = pd.read_excel('{}/Residues cost supply.xls'.format(base_dir), sheet_name="Forest residues traveled distan", skiprows=1,
                      index_col=0,header=0,squeeze=True).drop('Unnamed: 1', axis=1).drop(index={'TR','LI'}).fillna(0).rename(index={'UK':'GB','EL ':'GR'}, columns=renameColsForest)
    forest_residues = forest_residues[forest_residues.index.notnull()].sort_index()
    forest_residues.columns = forest_residues.columns.str.replace(r'<|>| km','',regex=True)

    agric_residues = pd.read_excel('{}/Residues cost supply.xls'.format(base_dir), sheet_name="Agri residues traveled distanc", skiprows=1,
                                   index_col=0,header=0,squeeze=True).drop('Unnamed: 1', axis=1).drop(index={'TR','LI'}).fillna(0).rename(index={'UK':'GB','EL ':'GR'}, columns=renameColsAgric)
    agric_residues = agric_residues[agric_residues.index.notnull()].sort_index()
    agric_residues.columns = agric_residues.columns.str.replace(r'<|>| km','',regex=True)

    #Convert PJ to MWh
    PJindex = ['forest residues','50','100','150','200','250','300','400']
    forest_residues[PJindex] = (forest_residues[PJindex] / 3.6 * 1e6).astype(int)
    PJindex = ['straw','50','100','150','200','250','300','400']
    agric_residues[PJindex] = (agric_residues[PJindex] / 3.6 * 1e6).astype(int)

    #Convert EUR/GJ to EUR/MWh
    Costindex = ['FixC','TransC']
    forest_residues[Costindex] = (forest_residues[Costindex] * 3.6).round(4)
    agric_residues[Costindex] = (agric_residues[Costindex] * 3.6).round(4)
    print(forest_residues.columns)


    # biomass_potentials = pd.read_csv('../resources/biomass_potentials.csv', index_col=0)
    #
    # # update forest residues and straw with new values
    # biomass_potentials['forest residues'].update(forest_residues['forest residues']+biomass_potentials['landscape care'])
    # biomass_potentials.drop('landscape care', inplace=True, axis=1)
    #
    # print(biomass_potentials)
    # biomass_potentials['straw'].update(agric_residues['straw'])
    # biomass_potentials.to_csv('../resources/biomass_potentials.csv') #snakemake.output.biomass_potentials)

    forest_residue_cost = forest_residues['FixC'] + forest_residues['TransC']*400
    agric_residue_cost = agric_residues['FixC'] + agric_residues['TransC']*400

    missing_countries = ['AL','BA','RS']
    for country in missing_countries:
        forest_residue_cost[country] = forest_residue_cost['BG']
        agric_residue_cost[country] = agric_residue_cost['BG']

    forest_residue_cost = forest_residue_cost.sort_index()
    agric_residue_cost = agric_residue_cost.sort_index()

    biomass_country_costs = pd.DataFrame({'forest residues' : forest_residue_cost, 'straw' : agric_residue_cost})

######## Read in data from ENSPRESO and add to biomass costs
    excel_out = pd.read_excel('{}/ENSPRESO_BIOMASS.xlsx'.format(base_dir), sheet_name="COST - NUTS0 EnergyCom",
                              index_col=[0, 1, 3, 2], header=0, squeeze=True).fillna(0).drop(columns={'Metada', 'Unnamed: 7', 'Units'})  # the summary sheet
    print(excel_out)
    # crops = ['Forestry energy residue',
    #          'Secondary forestry residues', 'Secondary Forestry residues sawdust',
    #          'Forestry residues from landscape care biomass', 'municipal biowaste',
    #          'manureslurry', 'sewage sludge']
    excel_out = excel_out.rename(index={'MINBIOAGRW1': 'straw',
                                        'MINBIOFRSR1': 'forest residues',
                                        'MINBIOWOOW1': 'industry wood residues',
                                        'MINBIOWOOW1a': 'industry wood residues sawdust',
                                        'MINBIOFRSR1a': 'forest residues landscape care',
                                        'MINBIOMUN1': 'municipal biowaste',
                                        'MINBIOGAS1': 'manureslurry',
                                        'MINBIOSLU1': 'sewage sludge'}).sort_index()

    year = 2040
    scenario = 'ENS_Ref'
    # biomass_country_costs = {}

    cropsToAdd = ['straw','forest residues','manureslurry','municipal biowaste','sewage sludge','industry wood residues']
    for crop in cropsToAdd:
        biomass_country_costs[crop] = (excel_out.loc[(year, scenario, crop)].rename(index={'UK':'GB','EL':'GR'})*3.6).round(4)
        # print(type(biomass_country_costs[crop].values))
        biomass_country_costs.rename(columns={'Cost': crop}, inplace=True)
    biomass_country_costs.index.name = None
    biomass_country_costs = biomass_country_costs.astype(float).round(4)

    # biomass_country_costs2 = pd.concat(biomass_country_costs, names=['type','countries'])#.sort_index()
    print(biomass_country_costs)
    # biomass_country_costs2.columns = ['biomass_type']
    # biomass_country_costs2.index.name = 'type'
    # print(biomass_country_costs2.index.name)
    # print(biomass_country_costs2.columns)
    # pivot_df = biomass_country_costs2.pivot(columns='type')#(index='countries', columns='type', values='Cost')
    # print(pivot_df)
    biomass_country_costs.to_csv('../resources/biomass_country_costs.csv')

if __name__ == "__main__":
    # if 'snakemake' not in globals():
    #     from helper import mock_snakemake
    #     snakemake = mock_snakemake('build_biomass_potentials')

    update_biomass_potentials()