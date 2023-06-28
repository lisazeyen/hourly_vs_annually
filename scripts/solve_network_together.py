
import pypsa, pandas as pd
from solve_network import (prepare_costs, palette, strip_network,
                           timescope, shutdown_lineexp, add_battery_constraints,
                           limit_resexp,set_co2_policy,
                           phase_outs, reduce_biomass_potential,
                           cost_parametrization, country_res_constraints,
                           average_every_nhours, add_unit_committment)
from resolve_network import (add_H2, add_dummies, res_constraints,
                             monthly_constraints, excess_constraints)


import logging
logger = logging.getLogger(__name__)
import sys

# Suppress logging of the slack bus choices
pypsa.pf.logger.setLevel(logging.WARNING)

from vresutils.benchmark import memory_logger
from _helpers import override_component_attrs


def solve_network(n, tech_palette):

    clean_techs, storage_techs, storage_chargers, storage_dischargers = palette(tech_palette)


    def extra_functionality(n, snapshots):

        add_battery_constraints(n)
        country_res_constraints(n, snakemake)

        if "res" in policy:
            logger.info("setting annual RES target")
            res_constraints(n, snakemake)
        if "monthly" in policy:
            logger.info("setting monthly RES target")
            monthly_constraints(n, snakemake)
        elif "exl" in policy:
            logger.info("setting excess limit on hourly matching")
            excess_constraints(n, snakemake)


    if snakemake.config["global"]["must_run"]:
        coal_i = n.links[n.links.carrier.isin(["lignite","coal"])].index
        n.links.loc[coal_i, "p_min_pu"] = 0.9
    n.consistency_check()

    # drop snapshots because of load shedding
    # to_drop = pd.Timestamp('2013-01-16 15:00:00')
    # new_snapshots = n.snapshots.drop(to_drop)
    # n.set_snapshots(new_snapshots)
    # final_sn = n.snapshots[n.snapshots<to_drop][-1]
    # n.snapshot_weightings.loc[final_sn] *= 2

    formulation = snakemake.config['solving']['options']['formulation']
    solver_options = snakemake.config['solving']['solver']
    solver_name = solver_options['name']
    solver_options["crossover"] = 0
    
    if snakemake.config["global"]["uc"]:
         add_unit_committment(n)
         
    linearized_uc = True if any(n.links.committable) else False


    
    # testing
    nhours = snakemake.config["scenario"]["temporal_resolution"]
    n = average_every_nhours(n, nhours)

    n.optimize(
           extra_functionality=extra_functionality,
           formulation=formulation,
           solver_name=solver_name,
           solver_options=solver_options,
           log_fn=snakemake.log.solver,
           linearized_unit_commitment=linearized_uc)

    # n.lopf(pyomo=False,
    #        extra_functionality=extra_functionality,
    #        formulation=formulation,
    #        solver_name=solver_name,
    #        solver_options=solver_options,
    #        solver_logfile=snakemake.log.solver)

#%%
if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('solve_network_together',
                                policy="exl1p0", palette='p1',
                                zone='DE', year='2025',
                                res_share="p0",
                                offtake_volume="3200",
                                storage="mtank")

    logging.basicConfig(filename=snakemake.log.python,
                    level=snakemake.config['logging_level'])

    #Wildcards & Settings ----------------------------------------------------
    policy = snakemake.wildcards.policy
    logger.info(f"solving network for policy: {policy}")

    tech_palette = snakemake.wildcards.palette
    logger.info(f"Technology palette: {tech_palette}")

    zone = snakemake.wildcards.zone
    logger.info(f"Bidding zone: {zone}")

    year = snakemake.wildcards.year
    logger.info(f"Year: {year}")

    area = snakemake.config['area']
    logger.info(f"Geoscope: {area}")

    res_share = float(snakemake.wildcards.res_share.replace("m","-").replace("p","."))
    if  snakemake.wildcards.res_share=="p0":
        res_share = timescope(zone, year, snakemake)["country_res_target"]
    logger.info(f"RES share: {res_share}")

    offtake_volume = float(snakemake.wildcards.offtake_volume)
    logger.info(f"H2 demand: {offtake_volume} MWh_h2/h")

    # import network -------------------------------------------------------
    n = pypsa.Network(timescope(zone, year, snakemake)['network_file'],
                      override_component_attrs=override_component_attrs())
    
    # adjust biomass CHP2 bus 2
    chp_i = n.links[n.links.carrier=="urban central solid biomass CHP"].index
    n.links.loc[chp_i, "bus2"] = ""
    remove_i = n.links[n.links.carrier.str.contains("biomass boiler")].index
    n.mremove("Link", remove_i)


    Nyears = 1 # years in simulation
    costs = prepare_costs(timescope(zone, year, snakemake)['costs_projection'],
                          snakemake.config['costs']['USD2013_to_EUR2013'],
                          snakemake.config['costs']['discountrate'],
                          Nyears,
                          snakemake.config['costs']['lifetime'],
                          year,
                          snakemake)

    strip_network(n, zone, area, snakemake)
    shutdown_lineexp(n)
    limit_resexp(n,year, snakemake)
    phase_outs(n, snakemake)
    reduce_biomass_potential(n)
    cost_parametrization(n, snakemake)
    set_co2_policy(n, snakemake, costs)

    add_H2(n, snakemake)
    add_dummies(n)


    with memory_logger(filename=getattr(snakemake.log, 'memory', None), interval=30.) as mem:

        solve_network(n, tech_palette)

        n.export_to_netcdf(snakemake.output.network)

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
