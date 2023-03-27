
import pypsa, numpy as np, pandas as pd
from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints
from vresutils.costdata import annuity

import logging
logger = logging.getLogger(__name__)
import sys

# Suppress logging of the slack bus choices
pypsa.pf.logger.setLevel(logging.WARNING)

from vresutils.benchmark import memory_logger
from _helpers import override_component_attrs


def palette(tech_palette):
    '''
    Define technology palette at CI node based on wildcard value
    '''

    if tech_palette == 'p1':
        clean_techs = ["onwind", "solar"]
        storage_techs = ["battery"]
        storage_chargers = ["battery charger"]
        storage_dischargers = ["battery discharger"]

    elif tech_palette == 'p2':
        clean_techs = ["onwind", "solar"]
        storage_techs = ["battery", "hydrogen"]
        storage_chargers = ["battery charger", "H2 Electrolysis"]
        storage_dischargers = ["battery discharger", "H2 Fuel Cell"]

    elif tech_palette == 'p3':
        clean_techs = ["onwind", "solar", "allam_ccs", "adv_geothermal"]  #"adv_nuclear", "adv_geothermal"
        storage_techs = ["battery", "hydrogen"]
        storage_chargers = ["battery charger", "H2 Electrolysis"]
        storage_dischargers = ["battery discharger", "H2 Fuel Cell"]

    else:
        logger.info(f"'palette' wildcard must be one of 'p1', 'p2' or 'p3'. Now is {tech_palette}.")
        sys.exit()

    return clean_techs, storage_techs, storage_chargers, storage_dischargers


def geoscope(n, zone, area):
    '''
    basenodes_to_keep -> geographical scope of the model
    country_nodes -> scope for national RES policy constraint
    node -> zone where C&I load is located
    '''
    def get_neighbours(zone):
        buses = {}
        for c in ["Line", "Link"]:
            items = ((n.df(c).bus0.str.contains(zone) | n.df(c).bus1.str.contains(zone))
                     & n.df(c).carrier.isin(["", "DC"]))
            buses[c] = n.df(c).loc[items, ["bus0", "bus1"]].values
        neighbors = pd.Index(np.concatenate((buses["Line"],
                                             buses["Link"])).ravel("K"))
        basenodes = n.buses[(n.buses.index.str[:2].isin(neighbors.str[:2]))
                            &(n.buses.carrier=="AC")].index
        return basenodes

    d = {}

    d['country_nodes'] = n.buses[(n.buses.index.str[:2].isin([zone]))
                                 & (n.buses.carrier=="AC")].index
    d['node'] = d['country_nodes'][:1][0]

    if area == 'EU':
        d['basenodes_to_keep'] = n.buses[n.buses.carrier=="AC"].index
    else:
        d['basenodes_to_keep'] = get_neighbours(zone)

    return d


def timescope(zone, year, snakemake):
    '''
    country_res_target -> value of national RES policy constraint for {year} and {zone}
    coal_phaseout -> countries that implement coal phase-out policy until {year}
    network_file -> input file with pypsa-eur-sec brownfield network for {year}
    costs_projection -> input file with technology costs for {year}
    '''

    d = {}

    d['country_res_target'] = snakemake.config[f'res_target_{year}'][f'{zone}']
    d['network_file'] = snakemake.input[f"network{year}"]
    d['costs_projection']  = snakemake.input[f"costs{year}"]

    return d


def cost_parametrization(n, snakemake):
    '''
    overwrite default price assumptions for primary energy carriers
    only for virtual generators located in 'EU {carrier}' buses
    '''

    for carrier in ['lignite', 'coal', 'gas']:
        n.generators.loc[n.generators.index.str.contains(f'EU {carrier}'), 'marginal_cost'] = snakemake.config['costs'][f'price_{carrier}']

    n.generators.loc[n.generators.carrier=="onwind", "marginal_cost"] = 0.015


def prepare_costs(cost_file, USD_to_EUR, discount_rate, Nyears, lifetime, year,
                  snakemake):

    #set all asset costs and other parameters
    costs = pd.read_csv(cost_file, index_col=[0,1]).sort_index()

    #correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.loc[costs.unit.str.contains("USD"), "value"] *= USD_to_EUR

    #min_count=1 is important to generate NaNs which are then filled by fillna
    costs = costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)
    costs = costs.fillna({"CO2 intensity" : 0,
                          "FOM" : 0,
                          "VOM" : 0,
                          "discount rate" : discount_rate,
                          "efficiency" : 1,
                          "fuel" : 0,
                          "investment" : 0,
                          "lifetime" : lifetime
    })

    # Advanced nuclear
    data_nuc = pd.DataFrame({'CO2 intensity': 0,
            'FOM': costs.loc['nuclear']['FOM'],
            'VOM': costs.loc['nuclear']['VOM'],
            'discount rate': discount_rate,
            'efficiency': 0.36,
            'fuel': costs.loc['nuclear']['fuel'],
            'investment': snakemake.config['costs']['adv_nuclear_overnight'] * 1e3 * snakemake.config['costs']['USD2021_to_EUR2021'],
            'lifetime': 40.0
            }, index=["adv_nuclear"])

    adv_geo_overnight = snakemake.config['costs'][f'adv_geo_overnight_{year}']
    allam_ccs_overnight = snakemake.config['costs'][f'allam_ccs_overnight_{year}']

    # Advanced geothermal
    data_geo = pd.DataFrame({'CO2 intensity': 0,
            'FOM': 0,
            'VOM': 0,
            'discount rate': discount_rate,
            'efficiency': 1,
            'fuel': 0,
            'investment':  adv_geo_overnight * 1e3 * 1,
            'lifetime': 30.0
            }, index=["adv_geothermal"])

    # Allam cycle ccs
    data_allam = pd.DataFrame({'CO2 intensity': 0,
            'FOM': 0, #%/year
            'FOM-abs' : 33000, #$/MW-yr
            'VOM': 3.2, #EUR/MWh
            'co2_seq': 40, #$/ton
            'discount rate': discount_rate,
            'efficiency': 0.54,
            'fuel': snakemake.config['costs']['price_gas'],
            'investment':  allam_ccs_overnight * 1e3 * 1,
            'lifetime': 30.0
            }, index=["allam_ccs"])

    costs = pd.concat([costs, data_nuc, data_geo, data_allam])

    annuity_factor = lambda v: annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100
    costs["fixed"] = [annuity_factor(v) * v["investment"] * Nyears for i, v in costs.iterrows()]

    return costs


def strip_network(n, zone, area, snakemake):
    # buses to keep
    nodes_to_keep = geoscope(n, zone, area)['basenodes_to_keep']
    new_nodes = pd.Index([bus + " " + suffix for bus in nodes_to_keep for
                          suffix in snakemake.config["node_suffixes_to_keep"]])
    nodes_to_keep = nodes_to_keep.union(new_nodes).union(snakemake.config['additional_nodes'])

    n.mremove('Bus', n.buses.index.symmetric_difference(nodes_to_keep))

    # make sure lines are kept
    n.lines.carrier = "AC"
    carrier_to_keep = snakemake.config['carrier_to_keep']

    for c in n.iterate_components(["Generator","Link","Line","Store","StorageUnit","Load"]):
        if c.name in ["Link","Line"]:
            location_boolean = c.df.bus0.isin(nodes_to_keep) & c.df.bus1.isin(nodes_to_keep)
        else:
            location_boolean = c.df.bus.isin(nodes_to_keep)
        to_keep = c.df.index[location_boolean & c.df.carrier.isin(carrier_to_keep)]
        to_drop = c.df.index.symmetric_difference(to_keep)
        n.mremove(c.name, to_drop)


def shutdown_lineexp(n):
    '''
    remove line expansion option
    '''
    n.lines.s_nom_extendable = False
    n.links.loc[n.links.carrier=='DC', 'p_nom_extendable'] = False


def limit_resexp(n, year, snakemake):
    '''
    limit expansion of renewable technologies per zone and carrier type
    as a ratio of max increase to 2021 capacity fleet
    (additional to zonal place availability constraint)
    '''
    ratio = snakemake.config['global'][f'limit_res_exp_{year}']

    fleet = n.generators.groupby([n.generators.bus.str[:2],
                                  n.generators.carrier]).p_nom.sum()
    fleet = fleet.rename(lambda x: x.split("-")[0], level=1).groupby(level=[0,1]).sum()
    ct_national_target = list(snakemake.config[f"res_target_{year}"].keys()) + ["EU"]

    fleet.drop(ct_national_target, errors="ignore", level=0, inplace=True)

    # allow build out of carriers which are not build yet
    fleet[fleet==0] = 1
    for ct, carrier in fleet.index:
        gen_i = ((n.generators.p_nom_extendable) & (n.generators.bus.str[:2]==ct)
                 & (n.generators.carrier.str.contains(carrier)))
        n.generators.loc[gen_i, "p_nom_max"] = ratio * fleet.loc[ct, carrier]


def phase_outs(n, snakemake):
    """Set planned phase outs of conventional powerplants.
    """
    phase_outs = snakemake.config["phase_out"]
    year = snakemake.wildcards.year
    for carrier in phase_outs.keys():
        countries = phase_outs[carrier][int(year)]
        carrier_list = [carrier]
        if carrier=="coal": carrier_list += ["lignite"]
        links_i = n.links.index.str[:2].isin(countries) & n.links.carrier.isin(carrier_list)
        n.links.loc[links_i, "p_nom"] = 0


def reduce_biomass_potential(n):
    '''
    remove solid biomass demand for industrial processes from overall biomass potential
    '''
    n.stores.loc[n.stores.index=='EU solid biomass', 'e_nom'] *= 0.45
    n.stores.loc[n.stores.index=='EU solid biomass', 'e_initial'] *= 0.45


def set_co2_policy(n, snakemake, costs):
    """Set CO2 policy to a cap or CO2 price.
    """
    gl_policy = snakemake.config['global']
    year = snakemake.wildcards.year

    if gl_policy['policy_type'] == "co2 cap":
        co2_cap = gl_policy['co2_share']*gl_policy['co2_baseline']
        logger.info(f"Setting global CO2 cap to {co2_cap}")
        n.global_constraints.at["CO2Limit","constant"] = co2_cap

    elif gl_policy['policy_type'] == "co2 price":
        n.global_constraints.drop("CO2Limit", inplace=True)
        co2_price = gl_policy[f'co2_price_{year}']
        logger.info(f"Setting CO2 price to {co2_price}")
        for carrier in ["coal", "oil", "gas", "lignite"]:
            n.generators.at[f"EU {carrier}","marginal_cost"] += co2_price*costs.at[carrier, 'CO2 intensity']


def freeze_capacities(n):

    for name, attr in [("generators","p"),("links","p"),("stores","e")]:
        df = getattr(n,name)
        df[attr + "_nom_extendable"] = False
        df[attr + "_nom"] = df[attr + "_nom_opt"]

    #allow more emissions
    n.stores.at["co2 atmosphere","e_nom"] *=2


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

def country_res_constraints(n, snakemake):

    ci_name = snakemake.config['ci']["name"]
    zone = snakemake.wildcards.zone
    year = snakemake.wildcards.year
    country_targets = snakemake.config[f"res_target_{year}"]


    for ct in country_targets.keys():

        if ct == zone:
            grid_buses = n.buses.index[(n.buses.index.str[:2]==ct) |
                                       (n.buses.index == f"{ci_name}")]
        else:
            grid_buses = n.buses.index[(n.buses.index.str[:2]==ct)]

        if grid_buses.empty: continue

        grid_res_techs = snakemake.config['global']['grid_res_techs']

        grid_loads = n.loads.index[n.loads.bus.isin(grid_buses)]

        country_res_gens = n.generators.index[n.generators.bus.isin(grid_buses)
                                              & n.generators.carrier.isin(grid_res_techs)]
        country_res_links = n.links.index[n.links.bus1.isin(grid_buses)
                                          & n.links.carrier.isin(grid_res_techs)]
        country_res_storage_units = n.storage_units.index[n.storage_units.bus.isin(grid_buses)
                                                          & n.storage_units.carrier.isin(grid_res_techs)]



        eff_links = n.links.loc[country_res_links, "efficiency"]

        weight_gens = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(country_res_gens)),
                                  index = n.snapshots,
                                  columns = country_res_gens)
        weight_links = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(country_res_links)),
                                  index = n.snapshots,
                                  columns = country_res_links).mul(eff_links)
        weight_sus= pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(country_res_storage_units)),
                                  index = n.snapshots,
                                  columns = country_res_storage_units)

        gens = linexpr((weight_gens, get_var(n, "Generator", "p")[country_res_gens]))
        links = linexpr((weight_links, get_var(n, "Link", "p")[country_res_links]))
        sus = linexpr((weight_sus, get_var(n, "StorageUnit", "p_dispatch")[country_res_storage_units]))
        lhs_temp = pd.concat([gens, links, sus], axis=1)

        lhs = join_exprs(lhs_temp)

        target = timescope(ct, year, snakemake)["country_res_target"]

        if  (snakemake.wildcards.res_share!="p0") and (ct == zone):
            target = float(snakemake.wildcards.res_share.replace("m","-").replace("p","."))


        total_load = (n.loads_t.p_set[grid_loads].sum(axis=1) * n.snapshot_weightings["generators"]).sum()

        # add for ct in zone electrolysis demand to load if not "reference" scenario
        # if (ct==zone) and (f"{ci_name}" in n.buses.index):

        #     logger.info("Consider electrolysis demand for RES target.")
        #     # H2 demand in zone
        #     offtake_volume = float(snakemake.wildcards.offtake_volume)
        #     # efficiency of electrolysis
        #     efficiency = n.links[n.links.carrier=="H2 Electrolysis"].efficiency.mean()

        #     # electricity demand of electrolysis
        #     demand_electrolysis = (offtake_volume/efficiency
        #                            *n.snapshot_weightings.generators).sum()
        #     total_load += demand_electrolysis

        print(f"country RES constraints for {ct} {target} and total load {total_load}")
        logger.info(f"country RES constraints for {ct} {target} and total load {total_load}")

        con = define_constraints(n, lhs, '=', target*total_load, f'countryRESconstraints_{ct}',f'countryREStarget_{ct}')


def solve_network(n, tech_palette):

    clean_techs, storage_techs, storage_chargers, storage_dischargers = palette(tech_palette)

    def extra_functionality(n, snapshots):

        add_battery_constraints(n)
        country_res_constraints(n, snakemake)


    if snakemake.config["global"]["must_run"]:
        coal_i = n.links[n.links.carrier.isin(["lignite","coal"])].index
        n.links.loc[coal_i, "p_min_pu"] = 0.9
    n.consistency_check()

    # drop snapshots because of load shedding
    to_drop = pd.Timestamp('2013-01-16 15:00:00')
    new_snapshots = n.snapshots.drop(to_drop)
    n.set_snapshots(new_snapshots)
    final_sn = n.snapshots[n.snapshots<to_drop][-1]
    n.snapshot_weightings.loc[final_sn] *= 2

    # # and another one
    # to_drop  = pd.Timestamp('2013-11-28 15:00:00')
    # new_snapshots = n.snapshots.drop(to_drop)
    # n.set_snapshots(new_snapshots)
    # final_sn = n.snapshots[n.snapshots<to_drop][-1]
    # n.snapshot_weightings.loc[final_sn] *= 2

    # and another one
    to_drop  = pd.Timestamp('2013-01-17 06:00:00')
    new_snapshots = n.snapshots.drop(to_drop)
    n.set_snapshots(new_snapshots)
    final_sn = n.snapshots[n.snapshots<to_drop][-1]
    n.snapshot_weightings.loc[final_sn] *= 2


    formulation = snakemake.config['solving']['options']['formulation']
    solver_options = snakemake.config['solving']['solver']
    solver_name = solver_options['name']
    solver_options["crossover"] = 0

    n.lopf(pyomo=False,
           extra_functionality=extra_functionality,
           formulation=formulation,
           solver_name=solver_name,
           solver_options=solver_options,
           solver_logfile=snakemake.log.solver)

    freeze_capacities(n)

    n.lopf(pyomo=False,
           formulation=formulation,
           solver_name=solver_name,
           solver_options=solver_options,
           solver_logfile=snakemake.log.solver)

#%%
if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('solve_base_network',
                                policy="ref", palette='p1', zone='DE', year='2025',
                                res_share="p0",
                                offtake_volume="3200")

    logging.basicConfig(filename=snakemake.log.python,
                    level=snakemake.config['logging_level'])

    #Wildcards & Settings ----------------------------------------------------
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

    # import network -------------------------------------------------------
    n = pypsa.Network(timescope(zone, year, snakemake)['network_file'],
                      override_component_attrs=override_component_attrs())

    Nyears = 1 # years in simulation
    costs = prepare_costs(timescope(zone, year, snakemake)['costs_projection'],
                          snakemake.config['costs']['USD2013_to_EUR2013'],
                          snakemake.config['costs']['discountrate'],
                          Nyears,
                          snakemake.config['costs']['lifetime'],
                          year,
                          snakemake)


    with memory_logger(filename=getattr(snakemake.log, 'memory', None), interval=30.) as mem:

        strip_network(n, zone, area, snakemake)
        shutdown_lineexp(n)
        limit_resexp(n,year, snakemake)
        phase_outs(n, snakemake)
        reduce_biomass_potential(n)
        cost_parametrization(n, snakemake)
        set_co2_policy(n, snakemake, costs)

        offtake_volume = float(snakemake.wildcards.offtake_volume)
        # efficiency of electrolysis
        efficiency = n.links[n.links.carrier=="H2 Electrolysis"].efficiency.mean()
        if snakemake.config["scenario"]["h2_demand_added"]:
            logger.info("Add electrolysis demand.")
            load_elec = offtake_volume/efficiency
            n.add("Load",
                  f"{geoscope(n,zone, area)['node']} electrolysis demand",
                  bus=geoscope(n,zone, area)['node'],
                  p_set=pd.Series(load_elec, index=n.snapshots),
                  carrier="electricity")

        solve_network(n, tech_palette)

        n.export_to_netcdf(snakemake.output.network)

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
