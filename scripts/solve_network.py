
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
        print(f"'palette' wildcard must be one of 'p1', 'p2' or 'p3'. Now is {tech_palette}.")
        sys.exit()

    return clean_techs, storage_techs, storage_chargers, storage_dischargers


def geoscope(zone, area):
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


def timescope(zone, year):
    '''
    country_res_target -> value of national RES policy constraint for {year} and {zone}
    coal_phaseout -> countries that implement coal phase-out policy until {year}
    network_file -> input file with pypsa-eur-sec brownfield network for {year}
    costs_projection -> input file with technology costs for {year}
    '''

    d = {}

    d['country_res_target'] = snakemake.config[f'res_target_{year}'][f'{zone}']
    d['coal_phaseout'] = snakemake.config[f'policy_{year}']
    d['network_file'] = snakemake.input[f"network{year}"]
    d['costs_projection']  = snakemake.input[f"costs{year}"]

    return d


def cost_parametrization(n):
    '''
    overwrite default price assumptions for primary energy carriers
    only for virtual generators located in 'EU {carrier}' buses
    '''

    for carrier in ['lignite', 'coal', 'gas']:
        n.generators.loc[n.generators.index.str.contains(f'EU {carrier}'), 'marginal_cost'] = snakemake.config['costs'][f'price_{carrier}']
    #n.generators[n.generators.index.str.contains('EU')].T
    # adjust wrongly set VOM of onshore wind
    n.generators.loc[n.generators.carrier=="onwind", "marginal_cost"] = 0.015


def prepare_costs(cost_file, USD_to_EUR, discount_rate, Nyears, lifetime, year):

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


def strip_network(n):
    # buses to keep
    nodes_to_keep = geoscope(zone, area)['basenodes_to_keep']
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


def limit_resexp(n, year):
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
        print(ct, carrier,ratio * fleet.loc[ct, carrier] )
        n.generators.loc[gen_i, "p_nom_max"] = ratio * fleet.loc[ct, carrier]

    for bus in fleet.bus.unique():
        if not bus[:2] in snakemake.config[f"res_target_{year}"].keys():
            for carrier in ['solar', 'onwind', 'offwind-ac', 'offwind-dc']:
                p_nom_fleet = fleet.loc[(fleet.bus == bus)
                                        & (fleet.carrier_s == carrier), "p_nom"].sum()
                #print(f'bus: {bus}, carrier: {carrier}' ,p_nom_fleet)
                n.generators.loc[(n.generators.p_nom_extendable==True) & (n.generators.bus == bus) & \
                                 (n.generators.carrier == carrier), "p_nom_max"] = ratio * p_nom_fleet


def nuclear_policy(n):
    '''
    remove nuclear PPs fleet for countries with nuclear ban policy
    '''
    for node in snakemake.config['nodes_with_nucsban']:
            n.links.loc[n.links['bus1'].str.contains(f'{node}') & (n.links.index.str.contains('nuclear')), 'p_nom'] = 0


def coal_policy(n):
    '''
    remove coal PPs fleet for countries with coal phase-out policy for {year}
    '''

    countries = timescope(zone, year)['coal_phaseout']

    for country in countries:
        n.links.loc[n.links['bus1'].str.contains(f'{country}') & (n.links.index.str.contains('coal')), 'p_nom'] = 0
        n.links.loc[n.links['bus1'].str.contains(f'{country}') & (n.links.index.str.contains('lignite')), 'p_nom'] = 0


def biomass_potential(n):
    '''
    remove solid biomass demand for industrial processes from overall biomass potential
    '''
    n.stores.loc[n.stores.index=='EU solid biomass', 'e_nom'] *= 0.45
    n.stores.loc[n.stores.index=='EU solid biomass', 'e_initial'] *= 0.45


def add_ci(n, policy, year):
    """Add C&I at its own node"""

    #first deal with global policy environment
    gl_policy = snakemake.config['global']

    if gl_policy['policy_type'] == "co2 cap":
        co2_cap = gl_policy['co2_share']*gl_policy['co2_baseline']
        print(f"Setting global CO2 cap to {co2_cap}")
        n.global_constraints.at["CO2Limit","constant"] = co2_cap

    elif gl_policy['policy_type'] == "co2 price":
        n.global_constraints.drop("CO2Limit", inplace=True)
        co2_price = gl_policy[f'co2_price_{year}']
        print(f"Setting CO2 price to {co2_price}")
        for carrier in ["coal", "oil", "gas", "lignite"]:
            n.generators.at[f"EU {carrier}","marginal_cost"] += co2_price*costs.at[carrier, 'CO2 intensity']


    #local C&I properties
    name = snakemake.config['ci']['name']
    #load = snakemake.config['ci']['load']
    ci_load = snakemake.config['ci_load'][f'{zone}']
    load = ci_load
    node = geoscope(zone, area)['node']


    if policy == "ref":
        return None

    #tech_palette options
    clean_techs, storage_techs, storage_chargers, storage_dischargers = palette(tech_palette)

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

    n.add("Load",
          f"{name} H2",
          carrier=f"{name} H2",
          bus=f"{name} H2",
          p_set=pd.Series(load,index=n.snapshots))

    #cost-less storage to indivcate flexible demand
    n.add("Store",
          f"{name} H2 Store",
          bus=f"{name} H2",
          e_cyclic=True,
          e_nom_extendable=True,
          carrier="H2 Store",
          capital_cost=0.001,#costs.at["hydrogen storage underground","fixed"],
          lifetime=costs.at["hydrogen storage underground","lifetime"],
          )

    if policy in ["res","cfe","exl","grf"]:
        n.add("Link",
              name + " export",
              bus0=name,
              bus1=node,
              marginal_cost=0.1, #large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
              p_nom=1e6)

    if policy in ["grd","res","grf"]:
        n.add("Link",
              name + " import",
              bus0=node,
              bus1=name,
              marginal_cost=0.001, #large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
              p_nom=1e6)

    if policy == "grd":
        return None


    #baseload clean energy generator
    if "green hydrogen OCGT" in clean_techs:
        n.add("Generator",
              name + " green hydrogen OCGT",
              carrier="green hydrogen OCGT",
              bus=name,
              p_nom_extendable = True,
              capital_cost=costs.at['OCGT', 'fixed'],
              marginal_cost=costs.at['OCGT', 'VOM']  + snakemake.config['costs']['price_green_hydrogen']/0.033/costs.at['OCGT', 'efficiency']) #hydrogen cost in EUR/kg, 0.033 MWhLHV/kg

    #baseload clean energy generator
    if "adv_nuclear" in clean_techs:
        n.add("Generator",
              f"{name} adv_nuclear",
              bus = name,
              carrier = 'nuclear',
              capital_cost = costs.loc['adv_nuclear']['fixed'],
              marginal_cost= costs.loc['adv_nuclear']['VOM']  + costs.loc['adv_nuclear']['fuel']/costs.loc['adv_nuclear']['efficiency'],
              p_nom_extendable = True,
              lifetime = costs.loc['adv_nuclear']['lifetime']
              )

    #baseload clean energy generator
    if "allam_ccs" in clean_techs:
        n.add("Generator",
              f"{name} allam_ccs",
              bus = name,
              carrier = 'gas',
              capital_cost = costs.loc['allam_ccs']['fixed'] + costs.loc['allam_ccs']['FOM-abs'],
              marginal_cost = costs.loc['allam_ccs']['VOM'] + \
                              costs.loc['allam_ccs']['fuel']/costs.loc['allam_ccs']['efficiency'] + \
                              costs.loc['allam_ccs']['co2_seq']*costs.at['gas', 'CO2 intensity']/costs.loc['allam_ccs']['efficiency'],
              p_nom_extendable = True,
              lifetime = costs.loc['allam_ccs']['lifetime'],
              efficiency = costs.loc['allam_ccs']['efficiency'],
              )

    #baseload clean energy generator
    if "adv_geothermal" in clean_techs:
        n.add("Generator",
              f"{name} adv_geothermal",
              bus = name,
              #carrier = '',
              capital_cost = costs.loc['adv_geothermal']['fixed'],
              marginal_cost= costs.loc['adv_geothermal']['VOM'],
              p_nom_extendable = True,
              lifetime = costs.loc['adv_geothermal']['lifetime']
              )

    #RES generator
    for carrier in ["onwind","solar"]:
        if carrier not in clean_techs:
            continue
        gen_template = node+" "+carrier+"-{}".format(year)
        n.add("Generator",
              f"{name} {carrier}",
              carrier=carrier,
              bus=name,
              p_nom_extendable=True,
              p_max_pu=n.generators_t.p_max_pu[gen_template],
              capital_cost=n.generators.at[gen_template,"capital_cost"],
              marginal_cost=n.generators.at[gen_template,"marginal_cost"])


    if "battery" in storage_techs:
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


def solve_network(n, policy, tech_palette):

    ci = snakemake.config['ci']
    name = ci['name']

    clean_techs, storage_techs, storage_chargers, storage_dischargers = palette(tech_palette)

    def res_constraints(n):

        res_gens = [name + " " + g for g in ci['res_techs']]

        weightings = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(res_gens)),
                                  index = n.snapshots,
                                  columns = res_gens)
        res = join_exprs(linexpr((weightings,get_var(n, "Generator", "p")[res_gens]))) # single line sum

        electrolysis = get_var(n, "Link", "p")[f"{name} H2 Electrolysis"]

        load = join_exprs(linexpr((-n.snapshot_weightings["generators"],electrolysis)))

        lhs = res + "\n" + load

        con = define_constraints(n, lhs, '=', 0., 'RESconstraints','REStarget')


    def excess_constraints(n):

        res_gens = [name + " " + g for g in ci['res_techs']]

        weightings = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(res_gens)),
                                  index = n.snapshots,
                                  columns = res_gens)
        res = join_exprs(linexpr((weightings,get_var(n, "Generator", "p")[res_gens]))) # single line sum

        electrolysis = get_var(n, "Link", "p")[f"{name} H2 Electrolysis"]

        allowed_excess = 1.2

        load = join_exprs(linexpr((-allowed_excess*n.snapshot_weightings["generators"],electrolysis)))

        lhs = res + "\n" + load

        con = define_constraints(n, lhs, '<=', 0., 'RESconstraints','REStarget')


    def country_res_constraints(n):

        country_targets = snakemake.config[f"res_target_{year}"]

        for ct in country_targets.keys():

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


            weigt_gens = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(country_res_gens)),
                                      index = n.snapshots,
                                      columns = country_res_gens)
            weigt_links = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(country_res_links)),
                                      index = n.snapshots,
                                      columns = country_res_links)
            weigt_sus= pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(country_res_storage_units)),
                                      index = n.snapshots,
                                      columns = country_res_storage_units)

            gens = linexpr((weigt_gens, get_var(n, "Generator", "p")[country_res_gens]))
            links = linexpr((weigt_links*n.links.loc[country_res_links, "efficiency"].values, get_var(n, "Link", "p")[country_res_links]))
            sus = linexpr((weigt_sus, get_var(n, "StorageUnit", "p_dispatch")[country_res_storage_units]))
            lhs_temp = pd.concat([gens, links, sus], axis=1)

            lhs = join_exprs(lhs_temp)

            target = timescope(ct, year)["country_res_target"]
            if ct == zone and snakemake.wildcards.res_share!="p0":
                target = res_share
            total_load = (n.loads_t.p_set[grid_loads].sum(axis=1)*n.snapshot_weightings["generators"]).sum() # number

            # add for ct in zone electrolysis demand to load
            if (ct==zone) and snakemake.config["scenario"]["h2_demand_included"]:
                logger.info("Consider electrolysis demand for RES target.")
                # H2 demand in zone
                offtake_volume = float(snakemake.wildcards.offtake_volume)
                # efficiency of electrolysis
                efficiency = n.links[n.links.carrier=="H2 Electrolysis"].efficiency.mean()
                # electricity demand of electrolysis
                demand_electrolysis = (offtake_volume/efficiency
                                       *n.snapshot_weightings.generators).sum()
                total_load += demand_electrolysis

            logger.info(f"country RES constraints for {ct} {target} and total load {total_load}")

            con = define_constraints(n, lhs, '=', target*total_load, f'countryRESconstraints_{ct}',f'countryREStarget_{ct}')


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


    def extra_functionality(n, snapshots):

        add_battery_constraints(n)
        country_res_constraints(n)

        if policy == "res":
            print("setting annual RES target")
            res_constraints(n)
        elif policy == "exl":
            print("setting excess limit on hourly matching")
            excess_constraints(n)
    if snakemake.config["global"]["must_run"]:
        coal_i = n.links[n.links.carrier.isin(["lignite","coal"])].index
        n.links.loc[coal_i, "p_min_pu"] = 0.9
    n.consistency_check()

    # drop snapshots because of load shedding
    to_drop = pd.Timestamp('2013-01-16 15:00:00')
    new_snapshots = n.snapshots.drop(pd.Timestamp('2013-01-16 15:00:00'))
    n.set_snapshots(new_snapshots)
    final_sn = n.snapshots[n.snapshots<to_drop][-1]
    n.snapshot_weightings.loc[final_sn] *= 2

    formulation = snakemake.config['solving']['options']['formulation']
    solver_options = snakemake.config['solving']['solver']
    solver_name = solver_options['name']


    n.lopf(pyomo=False,
           extra_functionality=extra_functionality,
           formulation=formulation,
           solver_name=solver_name,
           solver_options=solver_options,
           solver_logfile=snakemake.log.solver)

    def freeze_capacities(n):

        for name, attr in [("generators","p"),("links","p"),("stores","e")]:
            df = getattr(n,name)
            df[attr + "_nom_extendable"] = False
            df[attr + "_nom"] = df[attr + "_nom_opt"]

        #allow more emissions
        n.stores.at["co2 atmosphere","e_nom"] *=2

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
                                res_share="p10",
                                offtake_volume="3200")

    logging.basicConfig(filename=snakemake.log.python,
                    level=snakemake.config['logging_level'])

    #Wildcards & Settings ----------------------------------------------------
    #to fix background "base" network, solve first without H2 DEMAND
    policy = "ref"

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
        res_share = timescope(zone, year)["country_res_target"]
    logger.info(f"RES share: {res_share}")

    # import network -------------------------------------------------------
    n = pypsa.Network(timescope(zone, year)['network_file'],
                      override_component_attrs=override_component_attrs())

    Nyears = 1 # years in simulation
    costs = prepare_costs(timescope(zone, year)['costs_projection'],
                          snakemake.config['costs']['USD2013_to_EUR2013'],
                          snakemake.config['costs']['discountrate'],
                          Nyears,
                          snakemake.config['costs']['lifetime'],
                          year)


    with memory_logger(filename=getattr(snakemake.log, 'memory', None), interval=30.) as mem:

        strip_network(n)

        shutdown_lineexp(n)
        limit_resexp(n,year)
        nuclear_policy(n)
        coal_policy(n)
        biomass_potential(n)
        cost_parametrization(n)

        add_ci(n, policy, year)

        solve_network(n, policy, tech_palette)

        n.export_to_netcdf(snakemake.output.network)

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
