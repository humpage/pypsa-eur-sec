"""Solve network."""

import pypsa

import numpy as np
import pandas as pd

from pypsa.linopt import get_var, linexpr, define_constraints, write_objective
from pypsa.linopf import network_lopf, ilopf, lookup
from pypsa.pf import get_switchable_as_dense as get_as_dense
from pypsa.descriptors import get_extendable_i, get_non_extendable_i

from vresutils.benchmark import memory_logger
from helper import override_component_attrs
from solve_network import *
from pypsa.descriptors import nominal_attrs

import logging
logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)



def add_land_use_constraint(n):

    if 'm' in snakemake.wildcards.clusters:
        _add_land_use_constraint_m(n)
    else:
        _add_land_use_constraint(n)


def _add_land_use_constraint(n):
    #warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in ['solar', 'onwind', 'offwind-ac', 'offwind-dc']:
        existing = n.generators.loc[n.generators.carrier==carrier,"p_nom"].groupby(n.generators.bus.map(n.buses.location)).sum()
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        n.generators.loc[existing.index,"p_nom_max"] -= existing

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def _add_land_use_constraint_m(n):
    # if generators clustering is lower than network clustering, land_use accounting is at generators clusters

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    grouping_years = snakemake.config["existing_capacities"]["grouping_years"]
    current_horizon = snakemake.wildcards.planning_horizons

    for carrier in ['solar', 'onwind', 'offwind-ac', 'offwind-dc']:

        existing = n.generators.loc[n.generators.carrier==carrier,"p_nom"]
        ind = list(set([i.split(sep=" ")[0] + ' ' + i.split(sep=" ")[1] for i in existing.index]))

        previous_years = [
            str(y) for y in
            planning_horizons + grouping_years
            if y < int(snakemake.wildcards.planning_horizons)
        ]

        for p_year in previous_years:
            ind2 = [i for i in ind if i + " " + carrier + "-" + p_year in existing.index]
            sel_current = [i + " " + carrier + "-" + current_horizon for i in ind2]
            sel_p_year = [i + " " + carrier + "-" + p_year for i in ind2]
            n.generators.loc[sel_current, "p_nom_max"] -= existing.loc[sel_p_year].rename(lambda x: x[:-4] + current_horizon)

    n.generators.p_nom_max.clip(lower=0, inplace=True)


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


def add_pipe_retrofit_constraint(n):
    """Add constraint for retrofitting existing CH4 pipelines to H2 pipelines."""

    gas_pipes_i = n.links.query("carrier == 'gas pipeline' and p_nom_extendable").index
    h2_retrofitted_i = n.links.query("carrier == 'H2 pipeline retrofitted' and p_nom_extendable").index

    if h2_retrofitted_i.empty or gas_pipes_i.empty: return

    link_p_nom = get_var(n, "Link", "p_nom")

    CH4_per_H2 = 1 / n.config["sector"]["H2_retrofit_capacity_per_CH4"]
    fr = "H2 pipeline retrofitted"
    to = "gas pipeline"

    pipe_capacity = n.links.loc[gas_pipes_i, 'p_nom'].rename(basename)

    lhs = linexpr(
        (CH4_per_H2, link_p_nom.loc[h2_retrofitted_i].rename(index=lambda x: x.replace(fr, to))),
        (1, link_p_nom.loc[gas_pipes_i])
    )

    lhs.rename(basename, inplace=True)
    define_constraints(n, lhs, "=", pipe_capacity, 'Link', 'pipe_retrofit')


def add_co2_sequestration_limit(n, sns):

    co2_stores = n.stores.loc[n.stores.carrier=='co2 stored'].index

    if co2_stores.empty or ('Store', 'e') not in n.variables.index:
        return

    vars_final_co2_stored = get_var(n, 'Store', 'e').loc[sns[-1], co2_stores]

    lhs = linexpr((1, vars_final_co2_stored)).sum()

    limit = n.config["sector"].get("co2_sequestration_potential", 200) * 1e6
    # for o in opts:
    #     if not "seq" in o: continue
    #     limit = float(o[o.find("seq")+3:]) * 1e6
    #     print(float(o[o.find("seq")+3:]))
    #     break

    opts = snakemake.wildcards.sector_opts.split('-')
    for o in opts:
        if "seq" in o:
            limit = float(o[o.find("seq") + 3:]) * 1e6

    print('CO2 sequestration limit: ', limit)

    name = 'co2_sequestration_limit'
    sense = "<="

    # n.add("GlobalConstraint", name, sense=sense, constant=limit,
    #       type=np.nan, carrier_attribute=np.nan)

    define_constraints(n, lhs, sense, limit, 'GlobalConstraint',
                       'mu', axes=pd.Index([name]), spec=name)

def add_biofuel_constraint(n):

    options = snakemake.wildcards.sector_opts.split('-')
    liquid_biofuel_limit = 0
    biofuel_constraint_type = 'Lt'
    for o in options:
        if "B" in o:
            liquid_biofuel_limit = o[o.find("B") + 1:o.find("B") + 4]
            liquid_biofuel_limit = float(liquid_biofuel_limit.replace("p", "."))
            if len(o) > 7:
                biofuel_constraint_type = o[o.find("B") + 6:o.find("B") + 8]

    print('Liq biofuel minimum constraint: ', liquid_biofuel_limit, ' ', type(liquid_biofuel_limit))

    biofuel_i = n.links.query('carrier == "biomass to liquid"|carrier == "biomass to liquid CC"').index
    biofuel_vars = get_var(n, "Link", "p").loc[:, biofuel_i]
    biofuel_vars_eta = n.links.query('carrier == "biomass to liquid"|carrier == "biomass to liquid CC"').efficiency

    napkership = n.loads.p_set.filter(regex='naphtha for industry|kerosene for aviation|agriculture machinery oil$|shipping oil$').sum() * len(n.snapshots)
    landtrans = n.loads_t.p_set.filter(regex='land transport oil$').sum().sum()

    total_oil_load = napkership+landtrans
    limit = liquid_biofuel_limit * total_oil_load

    lhs = linexpr((biofuel_vars_eta, biofuel_vars)).sum().sum()

    name = 'liquid_biofuel_min'
    sense = '>='
    if biofuel_constraint_type == 'Eq':
        sense = '=='
    elif biofuel_constraint_type == 'Lt':
        sense = '>='

    define_constraints(n, lhs, sense, limit, 'Link', spec=name)


def add_msw_full_usage(n):
    print('Adding MSW incineration constraint')
    wasteCHP_i = n.links.carrier.filter(regex='waste CHP').index
    wasteCHP_vars = get_var(n, "Link", "p").loc[:, wasteCHP_i]

    lhs = linexpr((1, wasteCHP_vars)).sum().sum()
    rhs = n.generators.p_nom.filter(regex='municipal solid waste').sum() * len(n.snapshots)
    name = 'msw_full_incineration'
    sense = '=='
    define_constraints(n, lhs, sense, rhs, 'Link', spec=name)

def to_regex(pattern):
    """[summary]
    """
    return "(" + ").*(".join(pattern.split(" ")) + ")"


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

    if snakemake.config['foresight'] == 'myopic':
        add_land_use_constraint(n)

    return n


def solve_network(n, config, solver_opts=False, opts='', **kwargs):
    if not solver_opts:
        solver_options = config['solving']['solver'].copy()
    else:
        solver_options = solver_opts
    solver_name = solver_options.pop('name')
    cf_solving = config['solving']['options']
    track_iterations = cf_solving.get('track_iterations', False)
    min_iterations = cf_solving.get('min_iterations', 4)
    max_iterations = cf_solving.get('max_iterations', 6)
    keep_shadowprices = cf_solving.get('keep_shadowprices', True)

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    if cf_solving.get('skip_iterations', False):
        status, termination_condition = network_lopf(n, solver_name=solver_name, solver_options=solver_options,
                     extra_functionality=extra_functionality,
                     keep_shadowprices=keep_shadowprices,
                     keep_references=True,
                     keep_files=True,
                     **kwargs)
    else:
        ilopf(n, solver_name=solver_name, solver_options=solver_options,
              track_iterations=track_iterations,
              min_iterations=min_iterations,
              max_iterations=max_iterations,
              extra_functionality=extra_functionality,
              keep_shadowprices=keep_shadowprices,
              keep_references=True,
              keep_files=True,
              **kwargs)
    return n, status, termination_condition


def extra_functionality(n, snapshots):

    add_battery_constraints(n)
    add_pipe_retrofit_constraint(n)
    add_co2_sequestration_limit(n, snapshots)

    options = snakemake.wildcards.sector_opts.split('-')
    for o in options:
        if "B" in o:
            print('adding biofuel constraints')
            add_biofuel_constraint(n)
            add_msw_full_usage(n)
        if 'CCL' in o:
            print('adding ccl constraints')
            add_ccl_constraints(n)
        if 'convCCL' in o:
            print('adding conventional ccl constraints')
            add_ccl_constraints_conventional(n)

    wc = snakemake.wildcards.mga_tech.split("-")
    print('wc', wc)
    var_type = wc[0]
    pattern = wc[1]
    sense = snakemake.wildcards.sense
    process_objective_wildcard(n, var_type, pattern, sense)

    if "CostMax" not in n.global_constraints.index:
        define_mga_constraint(n, snapshots)
    define_mga_objective(n)

def process_objective_wildcard(n, var_type, pattern, sense):
    # """[summary]
    """[summary]

    Parameters
    ----------
    n : pypsa.Network
    mga_obj : list-like
        [var_type, pattern, sense]
    """

    mga_obj = [var_type, pattern, sense]

    # lookup = {
    #     "Line": ["Line"],
    #     "Transmission": ["Link", "Line"],
    # }
    # if var_type in lookup.keys():
    #     mga_obj[0] = lookup[var_type] #lookup[mga_obj[0]]
    #     mga_obj[1] = transmission_countries_to_index(n, pattern, var_type)

    lookup = {"max": -1, "min": 1}
    mga_obj[2] = lookup[sense]

    # attach to network
    n.mga_obj = mga_obj

    # print mga_obj to console
    print(mga_obj)


def define_mga_constraint(n, snapshots, epsilon=None, with_fix=None):
    """Build constraint defining near-optimal feasible space

    Parameters
    ----------
    n : pypsa.Network
    snapshots : Series|list-like
        snapshots
    epsilon : float, optional
        Allowed added cost compared to least-cost solution, by default None
    with_fix : bool, optional
        Calculation of allowed cost penalty should include cost of non-extendable components, by default None
    """

    if epsilon is None:
        epsilon = float(snakemake.wildcards.epsilon)
    print('epsilon: ', epsilon)

    if with_fix is None:
        with_fix = snakemake.config.get("include_non_extendable", True)

    expr = []

    print('snapshot weightings: ', n.snapshot_weightings)
    # operation
    for c, attr in lookup.query("marginal_cost").index:
        cost = (
            get_as_dense(n, c, "marginal_cost", snapshots)
            .loc[:, lambda ds: (ds != 0).all()]
            .mul(n.snapshot_weightings.generators, axis=0)
        )
        if cost.empty:
            continue
        expr.append(linexpr((cost, get_var(n, c, attr).loc[snapshots, cost.columns])).stack())

    # investment
    for c, attr in nominal_attrs.items():
        cost = n.df(c)["capital_cost"][get_extendable_i(n, c)]
        if cost.empty:
            continue
        expr.append(linexpr((cost, get_var(n, c, attr)[cost.index])))

    lhs = pd.concat(expr).sum()

    #n.objective_constant?
    if with_fix:
        ext_const = objective_constant(n, ext=True, nonext=False)
        nonext_const = objective_constant(n, ext=False, nonext=True)
        limit = (1 + epsilon) * (n.objective + ext_const + nonext_const) - nonext_const
    else:
        ext_const = objective_constant(n)
        limit = (1 + epsilon) * (n.objective + ext_const)

    name = 'CostMax'
    sense = '<='

    n.add("GlobalConstraint", name, sense=sense, constant=limit,
          type=np.nan, carrier_attribute=np.nan)

    define_constraints(n, lhs, sense, limit, "GlobalConstraint", 'mu_epsilon', spec=name)


def objective_constant(n, ext=True, nonext=True):
    """[summary]
    """

    if not (ext or nonext):
        return 0.0

    constant = 0.0
    for c, attr in nominal_attrs.items():
        i = pd.Index([])
        if ext:
            i = i.append(get_extendable_i(n, c))
        if nonext:
            i = i.append(get_non_extendable_i(n, c))
        constant += n.df(c)[attr][i] @ n.df(c).capital_cost[i]

    return constant


def define_mga_objective(n):

    components, pattern, sense = n.mga_obj

    pattern2 = []

    #Avoid capturing also CCGT when choosing CC as objective
    if pattern == 'CC':
        pattern = 'CC$'
    # Avoid capturing also CCGT and process emissions when choosing BECC as objective (caution: includes fossil CC)
    if pattern == 'BECC':
        pattern = '^((?!process emissions).)*CC(?!GT)'
    elif pattern == 'VRE':
        pattern = 'solar|wind'
    elif pattern == 'biofuels':
        pattern = 'biomass to liquid|electrobiofuels'

    # elif pattern == 'biomass CHP':
    #     pattern2 = 'gas CHP'
    # elif pattern == 'biomass industry':
    #     pattern2 = 'gas industry'

    print('pattern: ', to_regex(pattern))
    # print('pattern2: ', to_regex(pattern2))

    if isinstance(components, str):
        components = [components]
    print('components: ', components)

    #If n.df(c).loc[variables.columns, 'bus0'] == 'EU gas'
    #add biogas + biogas + CC + SNG + SNG + CC divided by other producers
    # bm = (df.loc['biogas'] + df.loc['biogas CC'] + df.loc['BioSNG'] + df.loc['BioSNG CC'])
    # allgasprod = bm + df.loc['gas'] + df.loc['Sabatier'] + df.loc['helmeth']
    # multiply by bm / allgasprod

    # .rename(columns='EU gas', inplace=True)
    #
    # bm_gas = get_var(n, 'Link', 'p').filter(regex='biogas$|biogas CC$|BioSNG$|BioSNG CC$')
    # other_gas = get_var(n, 'Link', 'p').filter(regex='Sabatier|helmeth')
    # fossil_gas = get_var(n, 'Generator', 'p').filter(regex='gas')
    #
    # bm_gas_expr = linexpr((1,bm_gas)).sum(axis=1)#.apply(join_exprs, axis=1)
    # print('bm_gas: ', bm_gas_expr)
    #
    # other_gas_expr = linexpr((1,other_gas)).sum(axis=1)#.apply(join_exprs, axis=1)
    # print('other_gas: ', other_gas_expr)
    #
    # fossil_gas_expr = linexpr((1,fossil_gas))#.apply(join_exprs, axis=1)
    # print('fossil_gas: ', fossil_gas_expr)
    #
    # denom = linexpr((1, bm_gas_expr), (1, 1/bm_gas_expr))#.apply(join_exprs, axis=1)#, (1, fossil_gas_expr))
    # print('denom: ', denom)
    #
    # bm_gas_expr = linexpr((bm_gas_expr, 1/denom))
    #
    # bm_share_gas = bm_gas_expr # * (other_gas_expr + fossil_gas_expr) #bm_gas / (bm_gas + other_gas)# + fossil_gas)
    #
    # print('bm_share_gas: ', bm_share_gas)

    terms = []
    for c in components:
        variables = get_var(n, c, 'p').filter(regex=to_regex(pattern))
        print(variables.head(50))
        variables.to_csv('temp.csv', mode='a')
        if c in ["Link", "Line"] and pattern in ["", "LN|LK", "LK|LN"]:
            coeffs = sense * n.df(c).loc[variables.index, "length"]
        elif pattern == 'CC$':
            coeffs = sense * n.df(c).loc[variables.columns, 'efficiency2']
        else:
            coeffs = sense

        # if not pattern2 == []:
        #     variables2 = get_var(n, c, 'p').filter(regex=to_regex(pattern2))
        #     print(variables2.head(50))
        #     variables2.to_csv('temp2.csv', mode='a')
        #
        #     terms.append(linexpr((bm_share_gas, variables2)))

    terms.append(linexpr((coeffs, variables)))

    joint_terms = pd.concat(terms)

    write_objective(n, joint_terms)

    # print objective to console
    print(joint_terms)


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'solve_network',
            simpl='',
            opts="",
            clusters="37",
            lv=1.0,
            sector_opts='168H-T-H-B-I-A-solar+p3-dist1',
            planning_horizons="2030",
        )

    logging.basicConfig(filename=snakemake.log.python,
                        level=snakemake.config['logging_level'])

    tmpdir = snakemake.config['solving'].get('tmpdir')
    if tmpdir is not None:
        from pathlib import Path
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    opts = snakemake.wildcards.opts.split('-')
    solve_opts = snakemake.config['solving']['options']

    fn = getattr(snakemake.log, 'memory', None)
    with memory_logger(filename=fn, interval=30.) as mem:

        overrides = override_component_attrs(snakemake.input.overrides)
        n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

        n = prepare_network(n, solve_opts)

        n, status, termination_condition = solve_network(n, config=snakemake.config, opts=opts,
                          solver_dir=tmpdir,
                          solver_logfile=snakemake.log.solver,
                          skip_objective=True)

        #read last line of solver log (gurobi output), check if last line does not contain 'Optimal objective
        # (or contains either 'Sub-optimal' or 'numerical trouble')
        #solution due to dicrepancy between gurobi solver status and termination_condition
        with open(snakemake.log.solver, "r") as file:
            last_line = file.readlines()[-2]

        if 'Optimal objective' not in last_line:
            solver_opts = snakemake.config['solving']['solver'].copy()
            solver_opts['ScaleFlag'] = 2
            # solver_opts['ObjScale'] = -0.5
            solver_opts['BarHomogeneous'] = 1
            solver_opts['NumericFocus'] = 2
            # solver_opts['Quad'] = 1
            # solver_opts['MarkowitzTol'] = 0.1
            print('Sub-optimal - rerunning with new solver settings: ', solver_opts)

            n, status, termination_condition = solve_network(n, config=snakemake.config, solver_opts=solver_opts, opts=opts,
                              solver_dir=tmpdir,
                              solver_logfile=snakemake.log.solver,
                              skip_objective=True)

        if "lv_limit" in n.global_constraints.index:
            n.line_volume_limit = n.global_constraints.at["lv_limit", "constant"]
            n.line_volume_limit_dual = n.global_constraints.at["lv_limit", "mu"]

        cluster = snakemake.wildcards.clusters
        lv = snakemake.wildcards.lv

        n.export_to_netcdf(snakemake.output[0])

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))