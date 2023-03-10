import pypsa
import pandas as pd
import numpy as np
from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints

import logging
logger = logging.getLogger(__name__)

# Suppress logging of the slack bus choices
pypsa.pf.logger.setLevel(logging.WARNING)

from vresutils.benchmark import memory_logger


from _helpers import override_component_attrs

from solve_network import geoscope, freeze_capacities, add_battery_constraints



def add_H2(n, snakemake):

    zone = snakemake.wildcards.zone
    area = snakemake.config['area']
    name = snakemake.config['ci']['name']
    policy = snakemake.wildcards.policy
    node = geoscope(n, zone, area)['node']
    year = snakemake.wildcards.year

    # remove electricity demand of electrolysis
    load_i = pd.Index([f"{node} electrolysis demand"])
    if load_i[0] in n.loads.index:
        n.mremove("Load", load_i)
    if policy == "ref":
        return None

    n.add("Bus",
          name)

    n.add("Bus",
          f"{name} H2",
          carrier="H2"
          )

    n.add("Link",
          f"{name} H2 Electrolysis",
          bus0=name,
          bus1=f"{name} H2",
          carrier="H2 Electrolysis",
          efficiency=n.links.at[f"{node} H2 Electrolysis"+"-{}".format(year), "efficiency"],
          capital_cost=n.links.at[f"{node} H2 Electrolysis"+"-{}".format(year), "capital_cost"],
          p_nom_extendable=True,
          lifetime=n.links.at[f"{node} H2 Electrolysis"+"-{}".format(year), "lifetime"]
          )

    # add offtake
    LHV_H2 = 33.33 # lower heating value [kWh/kg_H2]
    # offtake_price = float(snakemake.wildcards.offtake_price) * LHV_H2
    offtake_volume = snakemake.wildcards.offtake_volume

    # logger.info("Add H2 offtake with offtake price {}".format(offtake_price))
    # n.add("Generator",
    #        f"{name} H2" + " offtake",
    #        bus=f"{name} H2",
    #        carrier="offtake H2",
    #        marginal_cost=offtake_price,
    #        p_nom=offtake_volume,
    #        p_nom_extendable=False,
    #        p_max_pu=0,
    #        p_min_pu=-1)

    n.add("Load",
          f"{name} H2",
          carrier=f"{name} H2",
          bus=f"{name} H2",
          p_set=float(offtake_volume),
          )

    # storage cost depending on wildcard
    store_type = snakemake.wildcards.storage
    if store_type != "nostore":
        store_cost = snakemake.config["global"]["H2_store_cost"][store_type][float(snakemake.wildcards.year)]
        n.add("Store",
        f"{name} H2 Store",
        bus=f"{name} H2",
        e_cyclic=True,
        e_nom_extendable=True,
        # e_nom=load*8760,
        carrier="H2 Store",
        capital_cost = store_cost,
		)

    if any([x in policy for x in ["res", "cfe", "exl", "monthly"]]):
        n.add("Link",
              name + " export",
              bus0=name,
              bus1=node,
              carrier="export",
              marginal_cost=0.1, #large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
              p_nom=1e6)

    if any([x in policy for x in ["res", "grd", "monthly"]]):
        n.add("Link",
              name + " import",
              carrier = "import",
              bus0=node,
              bus1=name,
              marginal_cost=0.001, #large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
              p_nom=1e6)

    if policy == "grd":
        return None

    #RES generator
    for carrier in ["onwind","solar"]:
        gen_template = node+" "+carrier+"-{}".format(year)
        n.add("Generator",
              f"{name} {carrier}",
              carrier=carrier,
              bus=name,
              p_nom_extendable=True,
              p_max_pu=n.generators_t.p_max_pu[gen_template],
              capital_cost=n.generators.at[gen_template,"capital_cost"],
              marginal_cost=n.generators.at[gen_template,"marginal_cost"])


    if "battery" in ["battery"]:
        n.add("Bus",
              f"{name} battery",
              carrier="battery"
              )

        n.add("Store",
              f"{name} battery",
              bus=f"{name} battery",
              e_cyclic=True,
              e_nom_extendable=True,
              carrier="battery",
              capital_cost=n.stores.at[f"{node} battery"+"-{}".format(year), "capital_cost"],
              lifetime=n.stores.at[f"{node} battery"+"-{}".format(year), "lifetime"]
              )

        n.add("Link",
              f"{name} battery charger",
              bus0=name,
              bus1=f"{name} battery",
              carrier="battery charger",
              efficiency=n.links.at[f"{node} battery charger"+"-{}".format(year), "efficiency"],
              capital_cost=n.links.at[f"{node} battery charger"+"-{}".format(year), "capital_cost"],
              p_nom_extendable=True,
              lifetime=n.links.at[f"{node} battery charger"+"-{}".format(year), "lifetime"]
              )

        n.add("Link",
              f"{name} battery discharger",
              bus0=f"{name} battery",
              bus1=name,
              carrier="battery discharger",
              efficiency=n.links.at[f"{node} battery discharger"+"-{}".format(year), "efficiency"],
              marginal_cost=n.links.at[f"{node} battery discharger"+"-{}".format(year), "marginal_cost"],
              p_nom_extendable=True,
              lifetime=n.links.at[f"{node} battery discharger"+"-{}".format(year), "lifetime"]
              )

def add_dummies(n):
    elec_buses = n.buses.index[n.buses.carrier == "AC"]

    logger.info("adding dummies to",elec_buses)
    n.madd("Generator",
            elec_buses + " dummy",
            bus=elec_buses,
            carrier="dummy",
            p_nom=1e3,
            marginal_cost=1e6)


def res_constraints(n, snakemake):
    print("set res constraint")

    ci = snakemake.config['ci']
    name = snakemake.config['ci']['name']
    policy = snakemake.wildcards.policy

    res_gens = [name + " " + g for g in ci['res_techs']]

    weightings = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(res_gens)),
                              index = n.snapshots,
                              columns = res_gens)
    res = join_exprs(linexpr((weightings,get_var(n, "Generator", "p")[res_gens]))) # single line sum

    electrolysis = get_var(n, "Link", "p")[f"{name} H2 Electrolysis"]

    load = join_exprs(linexpr((-n.snapshot_weightings["generators"],electrolysis)))

    lhs = res + "\n" + load

    con = define_constraints(n, lhs, '>=', 0., 'RESconstraints','REStarget')

    allowed_excess = float(policy.replace("res","").replace("p","."))
    load = join_exprs(linexpr((-allowed_excess * n.snapshot_weightings["generators"],electrolysis)))

    lhs = res + "\n" + load


    con = define_constraints(n, lhs, '<=', 0., 'RESconstraints_excess',
                             'REStarget_excess')


def monthly_constraints(n, snakemake):


    ci = snakemake.config['ci']
    name = ci['name']
    policy = snakemake.wildcards.policy

    res_gens = [name + " " + g for g in ci['res_techs']]

    weightings = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(res_gens)),
                              index = n.snapshots,
                              columns = res_gens)
    res = linexpr((weightings,get_var(n, "Generator", "p")[res_gens])).sum(axis=1) # single line sum
    res = res.groupby(res.index.month).sum()

    electrolysis = get_var(n, "Link", "p")[f"{name} H2 Electrolysis"]

    # allowed_excess = float(policy.replace("monthly","").replace("p","."))
    allowed_excess = 1
    load = linexpr((-allowed_excess * n.snapshot_weightings["generators"],electrolysis))

    load = load.groupby(load.index.month).sum()

    for i in range(len(res.index)):
        lhs = res.iloc[i] + "\n" + load.iloc[i]

        con = define_constraints(n, lhs, '==', 0., f'RESconstraints_{i}',f'REStarget_{i}')

def excess_constraints(n, snakemake):

    res_gens = [f"{name} onwind",f"{name} solar"]

    policy = snakemake.wildcards.policy

    weightings = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(res_gens)),
                              index = n.snapshots,
                              columns = res_gens)
    res = join_exprs(linexpr((weightings,get_var(n, "Generator", "p")[res_gens]))) # single line sum

    electrolysis = get_var(n, "Link", "p")[f"{name} H2 Electrolysis"]

    load = join_exprs(linexpr((-n.snapshot_weightings["generators"],electrolysis)))

    lhs = res + "\n" + load

    con = define_constraints(n, lhs, '>=', 0., 'RESconstraints','REStarget')

    allowed_excess = float(policy.replace("exl","").replace("p","."))

    load = join_exprs(linexpr((-allowed_excess*n.snapshot_weightings["generators"],electrolysis)))

    lhs = res + "\n" + load

    con = define_constraints(n, lhs, '<=', 0., 'RESconstraints_excess','REStarget_excess')

def solve(policy, n):

    def extra_functionality(n, snapshots):

        add_battery_constraints(n)

        if "res" in policy:
            logger.info("setting annual RES target")
            res_constraints(n, snakemake)
        if "monthly" in policy:
            logger.info("setting monthly RES target")
            monthly_constraints(n, snakemake)
        elif "exl" in policy:
            logger.info("setting excess limit on hourly matching")
            excess_constraints(n, snakemake)

    fn = getattr(snakemake.log, 'memory', None)

    solver_options = snakemake.config["solving"]["solver"]
    solver_name = solver_options["name"]
    result, message = n.lopf(pyomo=False,
           extra_functionality=extra_functionality,
           solver_name=solver_name,
           solver_options=solver_options,
           solver_logfile=snakemake.log.solver)

    # if result != "ok" or message != "optimal":
    #     logger.info(f"solver ended with {result} and {message}, so quitting")
    #     sys.exit()



    return n


#%%
if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('resolve_network',
                                policy="res1p0", palette='p1', zone='DE',
                                year='2025',
                                res_share="p0",
                                offtake_volume="3200",
                                storage="nostore")

    logging.basicConfig(filename=snakemake.log.python,
                    level=snakemake.config['logging_level'])

    n = pypsa.Network(snakemake.input.base_network,
                      override_component_attrs=override_component_attrs())


    policy = snakemake.wildcards.policy
    logger.info(f"solving network for policy: {policy}")

    name = snakemake.config['ci']['name']

    zone = snakemake.wildcards.zone
    logger.info(f"solving network for bidding zone: {zone}")

    year = snakemake.wildcards.year
    logger.info(f"solving network year: {year}")

    area = snakemake.config['area']
    logger.info(f"solving with geoscope: {area}")

    node = geoscope(n, zone, area)['node']
    logger.info(f"solving with node: {node}")

    freeze_capacities(n)

    add_H2(n, snakemake)

    add_dummies(n)

    with memory_logger(filename=getattr(snakemake.log, 'memory', None), interval=30.) as mem:

        n = solve(policy, n)
        n.export_to_netcdf(snakemake.output.network)

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
