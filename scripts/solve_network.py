"""Solve network."""

import pypsa

import numpy as np

from pypsa.linopt import get_var, linexpr, define_constraints

from pypsa.linopf import network_lopf, ilopf

from vresutils.benchmark import memory_logger

from helper import override_component_attrs

import logging
logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


def add_land_use_constraint(n):

    #warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'
    for carrier in ['solar', 'onwind', 'offwind-ac', 'offwind-dc']:
        existing = n.generators.loc[n.generators.carrier == carrier, "p_nom"].groupby(n.generators.bus.map(n.buses.location)).sum()
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
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

    if snakemake.config['foresight'] == 'myopic':
        add_land_use_constraint(n)

    return n

# <<<<<<< HEAD
# def add_opts_constraints(n, opts=None):
#     if opts is None:
#         opts = snakemake.wildcards.opts.split('-')
#
#     if 'BAU' in opts:
#         mincaps = snakemake.config['electricity']['BAU_mincapacities']
#         def bau_mincapacities_rule(model, carrier):
#             gens = n.generators.index[n.generators.p_nom_extendable & (n.generators.carrier == carrier)]
#             return sum(model.generator_p_nom[gen] for gen in gens) >= mincaps[carrier]
#         n.model.bau_mincapacities = pypsa.opt.Constraint(list(mincaps), rule=bau_mincapacities_rule)
#
#     if 'SAFE' in opts:
#         peakdemand = (1. + snakemake.config['electricity']['SAFE_reservemargin']) * n.loads_t.p_set.sum(axis=1).max()
#         conv_techs = snakemake.config['plotting']['conv_techs']
#         exist_conv_caps = n.generators.loc[n.generators.carrier.isin(conv_techs) & ~n.generators.p_nom_extendable, 'p_nom'].sum()
#         ext_gens_i = n.generators.index[n.generators.carrier.isin(conv_techs) & n.generators.p_nom_extendable]
#         n.model.safe_peakdemand = pypsa.opt.Constraint(expr=sum(n.model.generator_p_nom[gen] for gen in ext_gens_i) >= peakdemand - exist_conv_caps)


def add_biofuel_constraint(n):

    opts = snakemake.wildcards.sector_opts.split('-')
    print('Options: ', opts)

    liquid_biofuel_limit = 0
    for o in opts:
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

# def add_eps_storage_constraint(n):
#     if not hasattr(n, 'epsilon'):
#         n.epsilon = 1e-5
#     fix_sus_i = n.storage_units.index[~ n.storage_units.p_nom_extendable]
#     n.model.objective.expr += sum(n.epsilon * n.model.state_of_charge[su, n.snapshots[0]] for su in fix_sus_i)
# =======
# >>>>>>> 87596dd015ab8f2fff8ef77881a1bd82a7255b14

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

    opts = snakemake.wildcards.sector_opts.split('-')
    for o in opts:
        if "B" in o:
            add_biofuel_constraint(n)


# def fix_branches(n, lines_s_nom=None, links_p_nom=None):
#     if lines_s_nom is not None and len(lines_s_nom) > 0:
#         n.lines.loc[lines_s_nom.index,"s_nom"] = lines_s_nom.values
#         n.lines.loc[lines_s_nom.index,"s_nom_extendable"] = False
#     if links_p_nom is not None and len(links_p_nom) > 0:
#         n.links.loc[links_p_nom.index,"p_nom"] = links_p_nom.values
#         print('Links p_nom: ',links_p_nom.values)
#         # n.links.loc[links_p_nom.index,"p_nom_extendable"] = True
#         n.links.loc[links_p_nom.index,"p_nom_extendable"] = False
def extra_functionality(n, snapshots):
    add_battery_constraints(n)


def solve_network(n, config, opts='', **kwargs):
    solver_options = config['solving']['solver'].copy()
    solver_name = solver_options.pop('name')
# <<<<<<< HEAD
#
#     def run_lopf(n, allow_warning_status=False, fix_zero_lines=False, fix_ext_lines=False):
#         free_output_series_dataframes(n)
#
#         if fix_zero_lines:
#             fix_lines_b = (n.lines.s_nom_opt == 0.) & n.lines.s_nom_extendable
#             fix_links_b = (n.links.carrier=='DC') & (n.links.p_nom_opt == 0.) & n.links.p_nom_extendable
#             fix_branches(n,
#                          lines_s_nom=pd.Series(0., n.lines.index[fix_lines_b]),
#                          links_p_nom=pd.Series(0., n.links.index[fix_links_b]))
#
#         if fix_ext_lines:
#             fix_branches(n,
#                          lines_s_nom=n.lines.loc[n.lines.s_nom_extendable, 's_nom_opt'],
#                          links_p_nom=n.links.loc[(n.links.carrier=='DC') & n.links.p_nom_extendable, 'p_nom_opt'])
#             if "line_volume_constraint" in n.global_constraints.index:
#                 n.global_constraints.drop("line_volume_constraint",inplace=True)
#         else:
#             if "line_volume_constraint" not in n.global_constraints.index:
#                 line_volume = getattr(n, 'line_volume_limit', None)
#                 if line_volume is not None and not np.isinf(line_volume):
#                     n.add("GlobalConstraint",
#                           "line_volume_constraint",
#                           type="transmission_volume_expansion_limit",
#                           carrier_attribute="AC,DC",
#                           sense="<=",
#                           constant=line_volume)
#
#
#         # Firing up solve will increase memory consumption tremendously, so
#         # make sure we freed everything we can
#         gc.collect()
#
#         #from pyomo.opt import ProblemFormat
#         #print("Saving model to MPS")
#         #n.model.write('/home/ka/ka_iai/ka_kc5996/projects/pypsa-eur/128-B-I.mps', format=ProblemFormat.mps)
#         #print("Model is saved to MPS")
#         #sys.exit()
#
#
#         status, termination_condition = n.lopf(pyomo=False,
#                                                solver_name=solver_name,
#                                                solver_logfile=solver_log,
#                                                solver_options=solver_options,
#                                                solver_dir=tmpdir,
#                                                extra_functionality=extra_functionality,
#                                                formulation=solve_opts['formulation'],
#                                                keep_shadowprices=True,
#                                                keep_references=True,
#                                                keep_files=True)
#                                                # extra_postprocessing=extra_postprocessing)
#                                                #keep_files=True
#                                                #free_memory={'pypsa'}
#
#         assert status == "ok" or allow_warning_status and status == 'warning', \
#             ("network_lopf did abort with status={} "
#              "and termination_condition={}"
#              .format(status, termination_condition))
#
#         if not fix_ext_lines and "line_volume_constraint" in n.global_constraints.index:
#             n.line_volume_limit_dual = n.global_constraints.at["line_volume_constraint","mu"]
#             print("line volume limit dual:",n.line_volume_limit_dual)
#
#         return status, termination_condition
#
#     lines_ext_b = n.lines.s_nom_extendable
#     if lines_ext_b.any():
#         # puh: ok, we need to iterate, since there is a relation
#         # between s/p_nom and r, x for branches.
#         msq_threshold = 0.01
#         lines = pd.DataFrame(n.lines[['r', 'x', 'type', 'num_parallel']])
#
#         lines['s_nom'] = (
#             np.sqrt(3) * n.lines['type'].map(n.line_types.i_nom) *
#             n.lines.bus0.map(n.buses.v_nom)
#         ).where(n.lines.type != '', n.lines['s_nom'])
#
#         lines_ext_typed_b = (n.lines.type != '') & lines_ext_b
#         lines_ext_untyped_b = (n.lines.type == '') & lines_ext_b
#
#         def update_line_parameters(n, zero_lines_below=10, fix_zero_lines=False):
#             if zero_lines_below > 0:
#                 n.lines.loc[n.lines.s_nom_opt < zero_lines_below, 's_nom_opt'] = 0.
#                 n.links.loc[(n.links.carrier=='DC') & (n.links.p_nom_opt < zero_lines_below), 'p_nom_opt'] = 0.
#
#             if lines_ext_untyped_b.any():
#                 for attr in ('r', 'x'):
#                     n.lines.loc[lines_ext_untyped_b, attr] = (
#                         lines[attr].multiply(lines['s_nom']/n.lines['s_nom_opt'])
#                     )
#
#             if lines_ext_typed_b.any():
#                 n.lines.loc[lines_ext_typed_b, 'num_parallel'] = (
#                     n.lines['s_nom_opt']/lines['s_nom']
#                 )
#                 logger.debug("lines.num_parallel={}".format(n.lines.loc[lines_ext_typed_b, 'num_parallel']))
#
#         iteration = 1
#
#         lines['s_nom_opt'] = lines['s_nom'] * n.lines['num_parallel'].where(n.lines.type != '', 1.)
#         status, termination_condition = run_lopf(n, allow_warning_status=True)
#
#         def msq_diff(n):
#             lines_err = np.sqrt(((n.lines['s_nom_opt'] - lines['s_nom_opt'])**2).mean())/lines['s_nom_opt'].mean()
#             logger.info("Mean square difference after iteration {} is {}".format(iteration, lines_err))
#             return lines_err
#
#         min_iterations = solve_opts.get('min_iterations', 2)
#         max_iterations = solve_opts.get('max_iterations', 999)
#
#         while msq_diff(n) > msq_threshold or iteration < min_iterations:
#             if iteration >= max_iterations:
#                 logger.info("Iteration {} beyond max_iterations {}. Stopping ...".format(iteration, max_iterations))
#                 break
#
#             update_line_parameters(n)
#             lines['s_nom_opt'] = n.lines['s_nom_opt']
#             iteration += 1
#
#             status, termination_condition = run_lopf(n, allow_warning_status=True)
#
#         update_line_parameters(n, zero_lines_below=100)
#
#         logger.info("Starting last run with fixed extendable lines")
#
#         # Not really needed, could also be taken out
#         # if 'snakemake' in globals():
#         #     fn = os.path.basename(snakemake.output[0])
#         #     n.export_to_netcdf('/home/vres/data/jonas/playground/pypsa-eur/' + fn)
#
#     # status, termination_condition = run_lopf(n, allow_warning_status=True, fix_ext_lines=True)
#     status, termination_condition = run_lopf(n, allow_warning_status=True, fix_ext_lines=False)
#
#     # Drop zero lines from network
#     # zero_lines_i = n.lines.index[(n.lines.s_nom_opt == 0.) & n.lines.s_nom_extendable]
#     # if len(zero_lines_i):
#     #     n.mremove("Line", zero_lines_i)
#     #     n.mremove("Line", zero_lines_i)
#     # zero_links_i = n.links.index[(n.links.p_nom_opt == 0.) & n.links.p_nom_extendable]
#     # if len(zero_links_i):
#     #     n.mremove("Link", zero_links_i)
#
#
# =======
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

        overrides = override_component_attrs(snakemake.input.overrides)
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
