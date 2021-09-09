"""Solve network."""

import pypsa

import numpy as np

import pandas as pd

from pypsa.linopt import get_var, linexpr, define_constraints

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

    # try:
    agg_p_nom_minmax = pd.read_csv(agg_p_nom_limits, index_col=list(range(2)))
    print(agg_p_nom_minmax)
    # except IOError:
    #     logger.exception("Need to specify the path to a .csv file containing "
    #                      "aggregate capacity limits per country in "
    #                      "config['electricity']['agg_p_nom_limit'].")
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
    minimum = agg_p_nom_minmax['min'].dropna()
    if not minimum.empty:
        define_constraints(n, p_nom_per_cc[minimum.index], '>=', minimum, 'agg_p_nom', 'min')
    maximum = agg_p_nom_minmax['max'].dropna()
    if not maximum.empty:
        define_constraints(n, p_nom_per_cc[maximum.index], '<=', maximum, 'agg_p_nom', 'max')


def add_biofuel_constraint(n):

    liquid_biofuel_limit = 0
    for o in n.opts:
        if "B" in o:
            liquid_biofuel_limit = o[o.find("B") + 1:o.find("B") + 4]
            liquid_biofuel_limit = float(liquid_biofuel_limit.replace("p", "."))

    print('Liq biofuel minimum constraint: ', liquid_biofuel_limit, ' ', type(liquid_biofuel_limit))

    biofuel_i = n.links.query('carrier == "biomass to liquid"').index
    biofuel_vars = get_var(n, "Link", "p").loc[:, biofuel_i]
    biofuel_vars_eta = n.links.query('carrier == "biomass to liquid"').efficiency

    napkership = n.loads.p_set.filter(regex='naphtha for industry|kerosene for aviation|oil for shipping').sum() * len(n.snapshots)
    landtrans = n.loads_t.p_set.filter(regex='land transport oil$').sum().sum()
    total_oil_load = napkership+landtrans
    liqfuelloadlimit = liquid_biofuel_limit * total_oil_load

    lhs = linexpr((biofuel_vars_eta, biofuel_vars)).sum().sum()
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
    add_battery_constraints(n)

    for o in n.opts:
        if "B" in o:
            print('adding biofuel constraints')
            add_biofuel_constraint(n)

    # if 'CCL' in n.opts:# and n.generators.p_nom_extendable: #.any():
    #     print('adding ccl constraints')
    #     add_ccl_constraints(n)


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
                     extra_functionality=extra_functionality, **kwargs)
    else:
        ilopf(n, solver_name=solver_name, solver_options=solver_options,
              track_iterations=track_iterations,
              min_iterations=min_iterations,
              max_iterations=max_iterations,
              extra_functionality=extra_functionality, **kwargs)
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

        n.export_to_netcdf(snakemake.output[0])

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
