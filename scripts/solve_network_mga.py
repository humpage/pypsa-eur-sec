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



def solve_network(n, config, opts='', **kwargs):
    solver_options = config['solving']['solver'].copy()
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
        network_lopf(n, solver_name=solver_name, solver_options=solver_options,
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
    return n


def extra_functionality(n, snapshots):
    #add_battery_constraints(n)
    #add_pipe_retrofit_constraint(n)
    #add_co2_sequestration_limit(n, snapshots)

    # options = snakemake.wildcards.sector_opts.split('-')
    # for o in options:
    #     if "B" in o:
    #         print('adding biofuel constraints')
    #         add_biofuel_constraint(n)
    #     if 'CCL' in o:
    #         print('adding ccl constraints')
    #         add_ccl_constraints(n)
    #     if 'convCCL' in o:
    #         print('adding conventional ccl constraints')
    #         add_ccl_constraints_conventional(n)

    #MGA
    # wc = snakemake.wildcards.objective.split("+")
    var_type = snakemake.wildcards.tech_type
    pattern = snakemake.wildcards.mga_tech
    sense = snakemake.wildcards.sense
    process_objective_wildcard(n, var_type, pattern, sense)
    define_mga_objective(n)
    define_mga_constraint(n, snapshots)


def process_objective_wildcard(n, var_type, pattern, sense):
    """[summary]

    Parameters
    ----------
    n : pypsa.Network
    mga_obj : list-like
        [var_type, pattern, sense]
    """

    mga_obj = [var_type, pattern, sense]

    lookup = {
        "Line": ["Line"],
        "Transmission": ["Link", "Line"],
    }
    if var_type in lookup.keys():
        mga_obj[0] = lookup[var_type] #lookup[mga_obj[0]]
        mga_obj[1] = transmission_countries_to_index(n, pattern, var_type)

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

    if with_fix is None:
        with_fix = snakemake.config.get("include_non_extendable", True)

    expr = []
    # print(8760. / len(n.snapshots))
    weight = 8760. / len(n.snapshots)
    # n.set_snapshots(n.snapshots[:nhours])
    # n.snapshot_weightings[:] = 8760. / len(n.snapshots)

    # print(n.snapshot_weightings.loc[snapshots])
    # print(snapshots)
    # operation
    for c, attr in lookup.query("marginal_cost").index:
        cost = (
            get_as_dense(n, c, "marginal_cost", snapshots)
            .loc[:, lambda ds: (ds != 0).all()]
            .mul(weight)#n.snapshot_weightings[snapshots], axis=0)
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

    if with_fix:
        ext_const = objective_constant(n, ext=True, nonext=False)
        nonext_const = objective_constant(n, ext=False, nonext=True)
        rhs = (1 + epsilon) * (n.objective + ext_const + nonext_const) - nonext_const
    else:
        ext_const = objective_constant(n)
        rhs = (1 + epsilon) * (n.objective + ext_const)

    define_constraints(n, lhs, "<=", rhs, "GlobalConstraint", "mu_epsilon")


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

    if isinstance(components, str):
        components = [components]
    print(components)

    terms = []
    for c in components:
        # print('TEST: ', c, nominal_attrs[c])
        # if c in ["StorageUnit"]:
        #     break

        variables = get_var(n, c, nominal_attrs[c]).filter(regex=to_regex(pattern))
        print(variables)

        if c in ["Link", "Line"] and pattern in ["", "LN|LK", "LK|LN"]:
            coeffs = sense * n.df(c).loc[variables.index, "length"]
        else:
            coeffs = sense

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

        n = solve_network(n, config=snakemake.config, opts=opts,
                          solver_dir=tmpdir,
                          solver_logfile=snakemake.log.solver)

        if "lv_limit" in n.global_constraints.index:
            n.line_volume_limit = n.global_constraints.at["lv_limit", "constant"]
            n.line_volume_limit_dual = n.global_constraints.at["lv_limit", "mu"]

        cluster = snakemake.wildcards.clusters
        lv = snakemake.wildcards.lv

        n.export_to_netcdf(snakemake.output[0])

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))