"""Solve network."""

import pypsa

import numpy as np

import os

import re

import pandas as pd

from pypsa.linopt import get_var, linexpr, define_constraints, get_dual

from pypsa.linopf import network_lopf, ilopf, join_exprs

from vresutils.benchmark import memory_logger

# from helper import override_component_attrs

import logging
logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)

#First tell PyPSA that links can have multiple outputs by
#overriding the component_attrs. This can be done for
#as many buses as you need with format busi for i = 2,3,4,5,....
#See https://pypsa.org/doc/components.html#link-with-multiple-outputs-or-inputs


override_component_attrs = pypsa.descriptors.Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
override_component_attrs["Link"].loc["bus2"] = ["string",np.nan,np.nan,"2nd bus","Input (optional)"]
override_component_attrs["Link"].loc["bus3"] = ["string",np.nan,np.nan,"3rd bus","Input (optional)"]
override_component_attrs["Link"].loc["bus4"] = ["string",np.nan,np.nan,"4th bus","Input (optional)"]
override_component_attrs["Link"].loc["efficiency2"] = ["static or series","per unit",1.,"2nd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["efficiency3"] = ["static or series","per unit",1.,"3rd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["efficiency4"] = ["static or series","per unit",1.,"4th bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["p2"] = ["series","MW",0.,"2nd bus output","Output"]
override_component_attrs["Link"].loc["p3"] = ["series","MW",0.,"3rd bus output","Output"]
override_component_attrs["Link"].loc["p4"] = ["series","MW",0.,"4th bus output","Output"]



def add_land_use_constraint(n):

    #warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'
    for carrier in ['solar', 'onwind', 'offwind-ac', 'offwind-dc']:
        existing = n.generators.loc[n.generators.carrier == carrier, "p_nom"].groupby(n.generators.bus.map(n.buses.location)).sum()
        print(existing)
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        print(existing.index)
        n.generators.loc[existing.index, "p_nom_max"] -= existing

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def prepare_network(n, solve_opts=None):
    
    if 'clip_p_max_pu' in solve_opts:
        for df in (n.generators_t.p_max_pu, n.generators_t.p_min_pu, n.storage_units_t.inflow):
            df.where(df>solve_opts['clip_p_max_pu'], other=0., inplace=True)

    if solve_opts.get('load_shedding'):
        n.add("Carrier", "Load")
        n.madd("Generator", n.buses.index, " load",
               bus=n.buses.index,
               carrier='load',
               sign=1e-3, # Adjust sign to measure p and p_nom in kW instead of MW
               marginal_cost=1e2, # Eur/kWh
               # intersect between macroeconomic and surveybased
               # willingness to pay
               # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
               p_nom=1e9 # kW
        )

    if solve_opts.get('noisy_costs'):
        for t in n.iterate_components():
            #if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if 'marginal_cost' in t.df:
                np.random.seed(174)
                t.df['marginal_cost'] += 1e-2 + 2e-3 * (np.random.random(len(t.df)) - 0.5)

        for t in n.iterate_components(['Line', 'Link']):
            np.random.seed(123)
            t.df['capital_cost'] += (1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)) * t.df['length']

    if solve_opts.get('nhours'):
        nhours = solve_opts['nhours']
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760./nhours

    # if snakemake.config['foresight'] == 'myopic':
    #     add_land_use_constraint(n)

    return n


def add_ccl_constraints(n):

    agg_p_nom_limits = n.config['existing_capacities'].get('agg_p_nom_limits')

    agg_p_nom_minmax = pd.read_csv(agg_p_nom_limits, index_col=list(range(2)))
    print(agg_p_nom_minmax)

    logger.info("Adding per carrier generation capacity constraints for individual countries")
    print('adding per carrier generation capacity constraints for individual countries')
    gen_country = n.generators.bus.map(n.buses.country)

    # cc means country and carrier
    p_nom_per_cc = (pd.DataFrame(
        {'p_nom': linexpr((1, get_var(n, 'Generator', 'p_nom'))),
         'country': gen_country, 'carrier': n.generators.carrier})
                    .dropna(subset=['p_nom'])
                    .groupby(['country', 'carrier']).p_nom
                    .apply(join_exprs))

    print('p_nom_per_cc: ', p_nom_per_cc)

    minimum = agg_p_nom_minmax['min'].dropna()
    if not minimum.empty:
        define_constraints(n, p_nom_per_cc[minimum.index], '>=', minimum, 'agg_p_nom', 'min')

    maximum = agg_p_nom_minmax['max'].dropna()
    if not maximum.empty:
        define_constraints(n, p_nom_per_cc[maximum.index], '<=', maximum, 'agg_p_nom', 'max')


def add_ccl_constraints_conventional(n):

    agg_p_nom_limits_conventional = n.config['existing_capacities'].get('agg_p_nom_limits_conventional')

    agg_p_nom_minmax_conventional = pd.read_csv(agg_p_nom_limits_conventional, index_col=list(range(2)))

    logger.info("Adding per carrier conventional link capacity constraints for individual countries")
    print('adding per carrier conventional link capacity constraints for individual countries')
    link_country = n.links.bus1.map(n.buses.country)

    # cc means country and carrier
    p_nom_per_cc_link = (pd.DataFrame(
        {'p_nom': linexpr((1, get_var(n, 'Link', 'p_nom'))),
         'country': link_country, 'carrier': n.links.carrier})
                    .dropna(subset=['p_nom'])
                    .groupby(['country', 'carrier']).p_nom
                    .apply(join_exprs))

    minimum_conventional = agg_p_nom_minmax_conventional['min'].dropna()
    if not minimum_conventional.empty:
        define_constraints(n, p_nom_per_cc_link[minimum_conventional.index], '>=', minimum_conventional, 'agg_p_nom', 'min')

    maximum_conventional = agg_p_nom_minmax_conventional['max'].dropna()
    if not maximum_conventional.empty:
        define_constraints(n, p_nom_per_cc_link[maximum_conventional.index], '<=', maximum_conventional, 'agg_p_nom', 'max')


def add_feasibility_constraints(n):
    #Find country specific initial load (el + industrial) from indata
    el_load = n.loads.p_set.filter(regex='0$') * 8760
    el_loadindustry = n.loads.p_set.filter(regex='industry electricity') * 8760
    totLoad = el_load + el_loadindustry
    #choose all solar and wind
    #n.generators.filter(regex='onwind')  # .sum(axis=1)
    #offwind = n.generators_t.p.filter(regex='offwind')  # .sum(axis=1)
    #solar = n.generators_t.p.filter(regex='solar')
    #Set all types to be lower than a factor times (year - 2020) times initial load (or initial load times a factor times (year-2020))
    # Carriers: onwind, offwind-ac, offwind-dc, solar
    #biofuel_i = n.generators.query('carrier == "biomass to liquid"').index
    #biofuel_vars = get_var(n, "Link", "p").loc[:, biofuel_i]

    gen_country = n.generators.bus.map(n.buses.country)

    # cc means country and carrier
    p_nom_per_cc = (pd.DataFrame(
        {'p_nom': linexpr((1, get_var(n, 'Generator', 'p'))),
         'country': gen_country, 'carrier': n.generators.carrier})
                    .dropna(subset=['p_nom'])
                    .groupby(['country', 'carrier']).p_nom
                    .apply(join_exprs))

    define_constraints(n, p_nom_per_cc, '<=', maximum_conventional, 'feasibility_p_nom', 'max')

def add_EQ_constraints(n, o, scaling=1e-1):
    float_regex = "[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == 'c':
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = n.snapshot_weightings.generators @ \
           n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    inflow = n.snapshot_weightings.stores @ \
             n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    inflow = inflow.reindex(load.index).fillna(0.)
    rhs = scaling * ( level * load - inflow )
    lhs_gen = linexpr((n.snapshot_weightings.generators * scaling,
                       get_var(n, "Generator", "p").T)
              ).T.groupby(ggrouper, axis=1).apply(join_exprs)
    lhs_spill = linexpr((-n.snapshot_weightings.stores * scaling,
                         get_var(n, "StorageUnit", "spill").T)
                ).T.groupby(sgrouper, axis=1).apply(join_exprs)
    lhs_spill = lhs_spill.reindex(lhs_gen.index).fillna("")
    lhs = lhs_gen + lhs_spill
    define_constraints(n, lhs, ">=", rhs, "equity", "min")



def add_biofuel_constraint(n):

    options = snakemake.wildcards.sector_opts.split('-')
    print('options: ', options)
    liquid_biofuel_limit = 0
    biofuel_constraint_type = 'Lt'
    for o in options:
        if "B" in o:
            liquid_biofuel_limit = o[o.find("B") + 1:o.find("B") + 4]
            liquid_biofuel_limit = float(liquid_biofuel_limit.replace("p", "."))
            print('Length of o: ', len(o))
            if len(o) > 7:
                biofuel_constraint_type = o[o.find("B") + 6:o.find("B") + 8]

    print('Liq biofuel minimum constraint: ', liquid_biofuel_limit, ' ', type(liquid_biofuel_limit))

    biofuel_i = n.links.query('carrier == "biomass to liquid"').index
    biofuel_vars = get_var(n, "Link", "p").loc[:, biofuel_i]
    print('Biofuel p', biofuel_vars)
    biofuel_vars_eta = n.links.query('carrier == "biomass to liquid"').efficiency
    print('Eta', biofuel_vars_eta)
    print('biofuel vars*eta', biofuel_vars*biofuel_vars_eta)

    napkership = n.loads.p_set.filter(regex='naphtha for industry|kerosene for aviation|shipping oil$').sum() * len(n.snapshots)
    print(napkership)
    landtrans = n.loads_t.p_set.filter(regex='land transport oil$').sum().sum()
    print(landtrans)

    total_oil_load = napkership+landtrans
    liqfuelloadlimit = liquid_biofuel_limit * total_oil_load

    lhs = linexpr((biofuel_vars_eta, biofuel_vars)).sum().sum()

    print('Constraint type: ', biofuel_constraint_type)

    if biofuel_constraint_type == 'Equ':
        define_constraints(n, lhs, "==", liqfuelloadlimit, 'Link', 'liquid_biofuel_min')
    elif biofuel_constraint_type == 'Lt':
        define_constraints(n, lhs, ">=", liqfuelloadlimit, 'Link', 'liquid_biofuel_min')


def add_battery_constraints(n):

    chargers_b = n.links.carrier.str.contains("battery charger")
    chargers = n.links.index[chargers_b & n.links.p_nom_extendable]
    dischargers = chargers.str.replace("charger", "discharger")

    if chargers.empty or ('Link', 'p_nom') not in n.variables.index:
        return

    link_p_nom = get_var(n, "Link", "p_nom")

    lhs = linexpr((1,link_p_nom[chargers]),
                  (-n.links.loc[dischargers, "efficiency"].values,
                   link_p_nom[dischargers].values))

    define_constraints(n, lhs, "=", 0, 'Link', 'charger_ratio')


def add_chp_constraints(n):

    electric_bool = (n.links.index.str.contains("urban central")
                     & n.links.index.str.contains("CHP")
                     & n.links.index.str.contains("electric"))
    heat_bool = (n.links.index.str.contains("urban central")
                 & n.links.index.str.contains("CHP")
                 & n.links.index.str.contains("heat"))

    electric = n.links.index[electric_bool]
    heat = n.links.index[heat_bool]

    electric_ext = n.links.index[electric_bool & n.links.p_nom_extendable]
    heat_ext = n.links.index[heat_bool & n.links.p_nom_extendable]

    electric_fix = n.links.index[electric_bool & ~n.links.p_nom_extendable]
    heat_fix = n.links.index[heat_bool & ~n.links.p_nom_extendable]

    link_p = get_var(n, "Link", "p")

    if not electric_ext.empty:

        link_p_nom = get_var(n, "Link", "p_nom")

        #ratio of output heat to electricity set by p_nom_ratio
        lhs = linexpr((n.links.loc[electric_ext, "efficiency"]
                       *n.links.loc[electric_ext, "p_nom_ratio"],
                       link_p_nom[electric_ext]),
                      (-n.links.loc[heat_ext, "efficiency"].values,
                       link_p_nom[heat_ext].values))

        define_constraints(n, lhs, "=", 0, 'chplink', 'fix_p_nom_ratio')

        #top_iso_fuel_line for extendable
        lhs = linexpr((1,link_p[heat_ext]),
                      (1,link_p[electric_ext].values),
                      (-1,link_p_nom[electric_ext].values))

        define_constraints(n, lhs, "<=", 0, 'chplink', 'top_iso_fuel_line_ext')

    if not electric_fix.empty:

        #top_iso_fuel_line for fixed
        lhs = linexpr((1,link_p[heat_fix]),
                      (1,link_p[electric_fix].values))

        rhs = n.links.loc[electric_fix, "p_nom"].values

        define_constraints(n, lhs, "<=", rhs, 'chplink', 'top_iso_fuel_line_fix')

    if not electric.empty:

        #backpressure
        lhs = linexpr((n.links.loc[electric, "c_b"].values
                       *n.links.loc[heat, "efficiency"],
                       link_p[heat]),
                      (-n.links.loc[electric, "efficiency"].values,
                       link_p[electric].values))

        define_constraints(n, lhs, "<=", 0, 'chplink', 'backpressure')


def extra_functionality(n, snapshots):
    print('adding extra constraints')

    options = snakemake.wildcards.sector_opts.split('-')
    print('options: ', options)
    for o in options:
        if "B" in o:
            print('adding biofuel constraints')
            add_biofuel_constraint(n)
        if 'CCL' in o:
            print('adding ccl constraints')
            add_ccl_constraints(n)
        if 'convCCL' in o:
            print('adding conventional ccl constraints')
            add_ccl_constraints_conventional(n)
        if 'EQ' in o:
            print('adding minimum supply constraints for each country')
            add_EQ_constraints(n, o)

    add_battery_constraints(n)


def solve_network(n, config, opts='', **kwargs):
    solver_options = config['solving']['solver'].copy()
    solver_name = solver_options.pop('name')

    cf_solving = config['solving']['options']
    track_iterations = cf_solving.get('track_iterations', False)
    min_iterations = cf_solving.get('min_iterations', 4)
    max_iterations = cf_solving.get('max_iterations', 6)

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    if cf_solving.get('skip_iterations', False):
        network_lopf(n, solver_name=solver_name, solver_options=solver_options,
                     extra_functionality=extra_functionality,
                     keep_shadowprices = True,
                     keep_references = True,
                     keep_files = True, **kwargs)
    else:
        ilopf(n, solver_name=solver_name, solver_options=solver_options,
              track_iterations=track_iterations,
              min_iterations=min_iterations,
              max_iterations=max_iterations,
              extra_functionality=extra_functionality,
              keep_shadowprices = True,
              keep_references = True,
              keep_files = True, **kwargs)
    return n


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'solve_network',
            simpl='',
            clusters=48,
            lv=1.0,
            sector_opts='Co2L0-168H-T-H-B-I-solar3-dist1',
            planning_horizons=2050,
        )

    logging.basicConfig(filename=snakemake.log.python,
                        level=snakemake.config['logging_level'])

    tmpdir = snakemake.config['solving'].get('tmpdir')
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    opts = snakemake.wildcards.opts.split('-')
    solve_opts = snakemake.config['solving']['options']

    fn = getattr(snakemake.log, 'memory', None)
    with memory_logger(filename=fn, interval=30.) as mem:

        overrides = override_component_attrs #(snakemake.input.overrides)
        n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

        n = prepare_network(n, solve_opts)

        n = solve_network(n, config=snakemake.config, opts=opts,
                          solver_dir=tmpdir,
                          solver_logfile=snakemake.log.solver)

        if "lv_limit" in n.global_constraints.index:
            n.line_volume_limit = n.global_constraints.at["lv_limit", "constant"]
            n.line_volume_limit_dual = n.global_constraints.at["lv_limit", "mu"]

        # print(n.constraints)
        # biofuel_constraint_dual = get_dual(n, 'Link', 'liquid_biofuel_min')
        # print(biofuel_constraint_dual)
        # print('Duals: ', n.duals)
        # n.duals.to_csv('results/testconstraints.csv','A')
        # print('Dual values: ', n.dualvalues)
        # n.constraints.to_csv('results/testconstraints.csv','A')
        # n.dualvalues.to_csv('results/testdualvalues.csv','A')
        # get_dual(n, 'Link', 'liquid_biofuel_min')
        # print(n.cons["Link"]["pnl"]["mu_upper"])
        # print(n.cons["Link"])#["liquid_biofuel_min"])

        cluster = snakemake.wildcards.clusters
        lv = snakemake.wildcards.lv

        sector_opts = snakemake.wildcards.sector_opts
        biofuel_sensitivity = snakemake.wildcards.biofuel_sensitivity
        electrofuel_sensitivity = snakemake.wildcards.electrofuel_sensitivity
        electrolysis_sensitivity = snakemake.wildcards.electrolysis_sensitivity
        cc_sensitivity = snakemake.wildcards.cc_sensitivity
        cs_sensitivity = snakemake.wildcards.cs_sensitivity
        oil_sensitivity = snakemake.wildcards.oil_sensitivity
        biomass_import_sensitivity = snakemake.wildcards.biomass_import_sensitivity
        planning_horizon = snakemake.wildcards.planning_horizons[-4:]
        headerBiofuelConstraint = '{}_lv{}_{}_{}_{}{}{}{}{}{}{}'.format(cluster,lv,sector_opts,
                                                                          planning_horizon,biofuel_sensitivity,
                                                                          electrofuel_sensitivity,
                                                                          electrolysis_sensitivity,
                                                                          cc_sensitivity,cs_sensitivity,
                                                                          oil_sensitivity,biomass_import_sensitivity)
        # print(n.cons["Link"]["df"]["liquid_biofuel_min"])

        biofuelConstraintFile = snakemake.config['results_dir'] + snakemake.config['run'] + '/biofuelminConstraint.csv'
        # print(biofuelConstraintFile)
        # if os.path.isfile(biofuelConstraintFile):
        #     print('File already exists')
        #     df = pd.read_csv(biofuelConstraintFile, index_col=0)
        # else:
        #     print('File does not exist')
        df = pd.DataFrame()
        #print(n.cons["Link"]["df"]["liquid_biofuel_min"][0])
        print(n.duals["Link"]["df"]["liquid_biofuel_min"])
        # data = pd.DataFrame({f'{headerBiofuelConstraint}':n.cons["Link"]["df"]["liquid_biofuel_min"].values}).T
        data = pd.DataFrame({f'{headerBiofuelConstraint}':n.duals["Link"]["df"]["liquid_biofuel_min"].values}).T
        # df[cluster][lv][sector_opts][planning_horizon][biofuel_sensitivity][electrofuel_sensitivity][electrolysis_sensitivity][cc_sensitivity][cs_sensitivity][oil_sensitivity][biomass_import_sensitivity] = n.cons["Link"]["df"]["liquid_biofuel_min"].values
        # df[headerBiofuelConstraint] = n.cons["Link"]["df"]["liquid_biofuel_min"]

        data.to_csv(biofuelConstraintFile, mode='a', header=False)#index='False')

        n.export_to_netcdf(snakemake.output[0])

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
