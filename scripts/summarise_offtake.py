#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:32:11 2022

@author: lisa
"""
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
from _helpers import override_component_attrs
from resolve_network import geoscope

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        import os
        os.chdir("/home/lisa/Documents/hourly_vs_annually/scripts")
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('summarise_offtake', palette='p1',
                                   zone='DE', year='2025',
                                   policy="offgrid", storage="nostore",
                                   offtake_volume=3200, res_share="p0")
        os.chdir("/home/lisa/Documents/hourly_vs_annually/")

LHV_H2 = 33.33 # lower heating value [kWh/kg_H2]

rename_techs = {"H2 Electrolysis": "electrolysis",
                "H2 Store": "H2 store",
                "dummy": "load shedding",
                "onwind": "onshore wind",
                "onwind local": "onshore wind local",
                "solar": "solar PV",
                "solar local": "solar PV local",
                "urban central solid biomass CHP": "biomass CHP",
                "offwind": "offshore wind",
                "offwind-dc": "offshore wind DC",
                "offwind-ac": "offshore wind AC",
                "import": "purchase",
                "export": "sale"}
#%%
def calculate_supply_energy(n, label, supply_energy):
    """calculate the total energy supply/consuption of each component at the buses aggregated by carrier"""


    bus_carriers = n.buses.carrier.unique()
    supply_energy = supply_energy.reindex(columns=supply_energy.columns.union(label))

    for i in bus_carriers:
        bus_map = (n.buses.carrier == i)
        # correct CHPs with missing bus2
        n.links.loc[n.links.carrier=="urban central solid biomass CHP", "bus2"] = ""
        bus_map.at[""] = False

        for c in n.iterate_components(n.one_port_components):

            items = c.df.index[c.df.bus.map(bus_map).fillna(False)]

            if len(items) == 0:
                continue

            s = c.pnl.p[items].multiply(n.snapshot_weightings.generators,axis=0).sum().multiply(c.df.loc[items, 'sign']).groupby(c.df.loc[items, 'carrier']).sum()
            s = pd.concat([s], keys=[c.list_name])
            s = pd.concat([s], keys=[i])

            supply_energy = supply_energy.reindex(s.index.union(supply_energy.index))
            supply_energy.loc[s.index, label] = s


        for c in n.iterate_components(n.branch_components):

            for end in [col[3:] for col in c.df.columns if col[:3] == "bus"]:

                items = c.df.index[c.df["bus" + str(end)].map(bus_map,
                                                              na_action="ignore")]

                if len(items) == 0:
                    continue

                s = (-1)*c.pnl["p"+end][items].multiply(n.snapshot_weightings.generators,axis=0).sum().groupby(c.df.loc[items, 'carrier']).sum()
                s.index = s.index + end
                s = pd.concat([s], keys=[c.list_name])
                s = pd.concat([s], keys=[i])

                supply_energy = supply_energy.reindex(s.index.union(supply_energy.index))

                supply_energy.loc[s.index, label] = s

    return supply_energy



def calculate_nodal_supply_energy(n, label, supply_energy):
    """calculate the total energy supply/consuption of each component at the buses aggregated by carrier"""


    bus_carriers = n.buses.carrier.unique()
    supply_energy = supply_energy.reindex(columns=supply_energy.columns.union(label))

    for i in bus_carriers:
        bus_map = (n.buses.carrier == i)
        # correct CHPs with missing bus2
        n.links.loc[n.links.carrier=="urban central solid biomass CHP", "bus2"] = ""
        bus_map.at[""] = False

        for c in n.iterate_components(n.one_port_components):

            items = c.df.index[c.df.bus.map(bus_map).fillna(False)]

            if len(items) == 0:
                continue

            s = c.pnl.p[items].multiply(n.snapshot_weightings.generators,axis=0).sum().multiply(c.df.loc[items, 'sign']).groupby([c.df.loc[items].index.str[:2], c.df.loc[items, "carrier"]]).sum()
            s = pd.concat([s], keys=[c.list_name])
            s = pd.concat([s], keys=[i])

            supply_energy = supply_energy.reindex(s.index.union(supply_energy.index))
            supply_energy.loc[s.index, label] = s


        for c in n.iterate_components(n.branch_components):

            for end in [col[3:] for col in c.df.columns if col[:3] == "bus"]:

                items = c.df.index[c.df["bus" + str(end)].map(bus_map,
                                                              na_action="ignore")]

                if len(items) == 0:
                    continue

                s = (-1)*c.pnl["p"+end][items].multiply(n.snapshot_weightings.generators,axis=0).sum().groupby([c.df.loc[items, ("bus"+end)].str[:2], c.df.loc[items, "carrier"]]).sum()
                s.rename(index=lambda x: x+end, level=1, inplace=True)
                s = pd.concat([s], keys=[c.list_name])
                s = pd.concat([s], keys=[i])

                supply_energy = supply_energy.reindex(s.index.union(supply_energy.index))

                supply_energy.loc[s.index, label] = s

    return supply_energy


def calculate_h2_generationmix(n, label, gen_mix):
    """calculate the generation mix used for H2 production"""


    # generation_mix = generation_mix.reindex(columns=generation_mix.columns.union(label))
    bus_map = (n.buses.carrier == "AC")
    # correct CHPs with missing bus2
    n.links.loc[n.links.carrier=="urban central solid biomass CHP", "bus2"] = ""
    bus_map.at[""] = False
    zone = snakemake.wildcards.zone
    ci = snakemake.config['ci']
    name = ci['name']

    generation_mix = pd.DataFrame()


    for c in n.iterate_components(n.one_port_components):

        items = c.df.index[c.df.bus.map(bus_map).fillna(False)]

        if len(items) == 0:
            continue

        s = (c.pnl.p[items].multiply(n.snapshot_weightings.generators,axis=0)
             .multiply(c.df.loc[items, 'sign'])
             .groupby([c.df.loc[items].index.str[:2], c.df.loc[items, "carrier"]],
                      axis=1).sum())
        s = s.reindex(columns=[zone, name[:2]], level=0)
        s = pd.concat([s], keys=[c.list_name], axis=1)

        generation_mix = generation_mix.reindex(s.index.union(generation_mix.index))
        generation_mix = generation_mix.reindex(columns= s.columns.union(generation_mix.columns))
        generation_mix.loc[s.index, s.columns] = s


    for c in n.iterate_components(n.branch_components):

        for end in [col[3:] for col in c.df.columns if col[:3] == "bus"]:

            items = c.df.index[c.df["bus" + str(end)].map(bus_map, na_action="ignore")]

            if len(items) == 0:
                continue

            s = ((-1)*c.pnl["p"+end][items].multiply(n.snapshot_weightings.generators,axis=0)
                 .groupby([c.df.loc[items, ("bus"+end)].str[:2], c.df.loc[items, "carrier"]], axis=1).sum())
            s.rename(columns=lambda x: x+end, level=1, inplace=True)
            s = s.reindex(columns=[zone, name[:2]], level=0)
            s = pd.concat([s], keys=[c.list_name], axis=1)

            generation_mix = generation_mix.reindex(s.index.union(generation_mix.index))
            generation_mix = generation_mix.reindex(columns= s.columns.union(generation_mix.columns))
            generation_mix.loc[s.index, s.columns] = s

    electrolysis = generation_mix["links"][name[:2]]["H2 Electrolysis0"]
    if "import1" in generation_mix.columns.levels[2]:
        imports = generation_mix["links"][name[:2]]["import1"]
    else:
        imports = 0

    if "export0" in generation_mix.columns.levels[2]:
        exports = generation_mix["links"][name[:2]]["export0"]
    else:
        exports = 0
    to_drop = ["AC0", "AC1", "DC0", "DC1", "import0", "export1", "electricity",
               "PHS", "hydro", 'battery charger0', 'battery discharger1',
               "H2 Electrolysis0", "H2 Fuel Cell1"]
    # generation mix in zone
    zone_generation = generation_mix.xs(zone, level=1, axis=1).drop(to_drop, level=1, axis=1, errors="ignore")
    total_generation = zone_generation[zone_generation>0].sum(axis=1)
    # share of each technology in zone for each time step
    share = zone_generation.div(total_generation, axis=0)
    # total imported electricity by carrier
    import_mix = share.mul(imports, axis=0)
    # local generation

    local_generation = generation_mix.loc[:,generation_mix.columns.get_level_values(1)==name[:2]].droplevel(1, axis=1)
    local_gens = local_generation.columns.get_level_values(1).isin(ci["res_techs"])
    if "battery charger0" in local_generation.columns.get_level_values(1).unique():
        battery_charged_by = local_generation.loc[round(local_generation[("links", "battery charger0")])==0, local_gens].sum()
        battery_c_share = battery_charged_by/battery_charged_by.sum()
    local_generation = local_generation.loc[:,local_generation.columns.get_level_values(1).isin(["onwind","solar", "battery discharger1"])]
    if "battery discharger1" in local_generation.columns.get_level_values(1).unique():
        for i in battery_c_share.index:
            local_generation[i] += local_generation[("links", "battery discharger1")] * battery_c_share.loc[i]
        local_generation.drop("battery discharger1", level=1, axis=1, inplace=True)
    local_generation.rename(columns=lambda x: x + " local", level=1, inplace=True)

    total = pd.concat([import_mix, local_generation], axis=1)

    tot_mix = total.div(total.sum(axis=1), axis=0).mul(abs(electrolysis), axis=0).sum()

    gen_mix = gen_mix.reindex(columns=gen_mix.columns.union(label))
    gen_mix = gen_mix.reindex(index=tot_mix.index)
    gen_mix.loc[tot_mix.index, label] = tot_mix

    return gen_mix

opt_name = {
    "Store": "e",
    "Line": "s",
    "Transformer": "s"
}


def assign_locations(n, name):
    for c in n.iterate_components(n.one_port_components|n.branch_components):
        c.df["location"] = c.df.rename(index=lambda x: f"{name}" if f"{name}" in x else "rest").index


def calculate_nodal_capacities(n, label, nodal_capacities):
    #Beware this also has extraneous locations for country (e.g. biomass) or continent-wide (e.g. fossil gas/oil) stuff
    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load"}):
        nodal_capacities_c = c.df.groupby(["location","carrier"])[opt_name.get(c.name,"p") + "_nom_opt"].sum()
        index = pd.MultiIndex.from_tuples([(c.list_name,) + t for t in nodal_capacities_c.index.to_list()])
        nodal_capacities = nodal_capacities.reindex(index.union(nodal_capacities.index))
        nodal_capacities = nodal_capacities.reindex(columns=nodal_capacities.columns.union(label))
        nodal_capacities.loc[index,label] = nodal_capacities_c.values

    return nodal_capacities

ls_dict = {"ref": "-",
           "cfe": ":",
           "exl": "--",
           "grd": "x-",
           "res": "o-",
           "offgrid": "v-"}
color_dict = {"ref": '#377eb8',
           "cfe": '#ff7f00',
           "exl": '#4daf4a',
           "grd":  '#f781bf',
           "res": '#a65628',
           "offgrid": '#984ea3'
    }


def calculate_weighted_prices(n, label, weighted_prices):
    # Warning: doesn't include storage units as loads

    weighted_prices = weighted_prices.reindex(pd.Index([
        "AC",
        "H2"
    ]))
    cols = pd.MultiIndex.from_product([label.levels[0], label.levels[1], label.levels[2], label.levels[3], [f"{name}", "rest"]])

    weighted_prices = weighted_prices.reindex(columns = weighted_prices.columns.union(cols))
    link_loads = {"AC":  ["battery charger", "H2 Electrolysis"],
                  "H2": ["H2 Fuel Cell"]}

    for carrier in link_loads:

        buses = n.buses.index[n.buses.carrier==carrier]

        if buses.empty:
            continue

        if carrier in ["H2", "gas"]:
            load = pd.DataFrame(index=n.snapshots, columns=buses, data=0.)
        else:
            load = n.loads_t.p_set.loc[:,n.loads.bus.map(n.buses.carrier)==carrier]

        for tech in link_loads[carrier]:

            names = n.links.index[n.links.index.to_series().str[-len(tech):] == tech]

            if names.empty:
                continue

            load = pd.concat([load, n.links_t.p0[names].groupby(n.links.loc[names, "bus0"],axis=1).sum()], axis=1)

        load = load.groupby(load.columns,axis=1).sum()
        if carrier == "AC":
            a = load/load.max() * n.buses_t.marginal_price.reindex(columns=load.columns)
        else:
            a = n.buses_t.marginal_price.loc[:,n.buses.carrier==carrier]
        load_rest = a.loc[:,~a.columns.str.contains(f"{name}")].mean(axis=1).mean()
        load_google = a.loc[:,a.columns.str.contains(f"{name}")].mean(axis=1).mean()

        weighted_prices.loc[carrier,(label[0][0], label[0][1], label[0][2], label[0][3], "rest")] = load_rest
        weighted_prices.loc[carrier,(label[0][0], label[0][1], label[0][2], label[0][3],f"{name}")]= load_google

    return weighted_prices


def calculate_curtailment(n, label, curtailment):
    res = ["onwind", "offwind-ac", "offwind-dc", "offwind", "solar"]
    curtailment_n = ((n.generators_t.p.sum().groupby([n.generators.carrier, n.generators.location]).sum())
   / ((n.generators_t.p_max_pu * n.generators.p_nom_opt).sum().groupby([n.generators.carrier, n.generators.location]).sum()))
    all_res_p = (n.generators_t.p.sum()
                 .groupby([n.generators.carrier, n.generators.location]).sum()
                 .reindex(res,level=0).xs("rest", level=1))
    all_res_max = ((n.generators_t.p_max_pu * n.generators.p_nom_opt).sum()
                   .groupby([n.generators.carrier, n.generators.location]).sum()
                   .reindex(res,level=0).xs("rest", level=1))
    curtailment_n.loc[("all", "all")] = all_res_p.sum()/all_res_max.sum()
    curtailment = curtailment.reindex(columns=curtailment.columns.union(label))
    curtailment = curtailment.reindex(index=curtailment.index.union(curtailment_n.index))
    curtailment.loc[curtailment_n.index, label] = curtailment_n
    return curtailment


def calculate_nodal_costs(n, label, nodal_costs):
    #Beware this also has extraneous locations for country (e.g. biomass) or continent-wide (e.g. fossil gas/oil) stuff
    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load"}):
        c.df["capital_costs"] = c.df.capital_cost * c.df[opt_name.get(c.name, "p") + "_nom_opt"]
        capital_costs = c.df.groupby(["location", "carrier"])["capital_costs"].sum()
        index = pd.MultiIndex.from_tuples([(c.list_name, "capital") + t for t in capital_costs.index.to_list()])
        nodal_costs = nodal_costs.reindex(index.union(nodal_costs.index))
        nodal_costs = nodal_costs.reindex(columns=nodal_costs.columns.union(label))
        nodal_costs.loc[index,label] = capital_costs.values

        if c.name == "Link":
            p = c.pnl.p0.multiply(n.snapshot_weightings.generators, axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
            p_all[p_all < 0.] = 0.
            p = p_all.sum()
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0).sum()

        #correct sequestration cost
        if c.name == "Store":
            items = c.df.index[(c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.)]
            c.df.loc[items, "marginal_cost"] = -20.

        c.df["marginal_costs"] = p*c.df.marginal_cost
        marginal_costs = c.df.groupby(["location", "carrier"])["marginal_costs"].sum()
        index = pd.MultiIndex.from_tuples([(c.list_name, "marginal") + t for t in marginal_costs.index.to_list()])
        nodal_costs = nodal_costs.reindex(index.union(nodal_costs.index))

        nodal_costs.loc[index, label] = marginal_costs.values

    return nodal_costs


def calculate_costs(n, label, costs):
    costs = costs.reindex(columns=costs.columns.union(label))
    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load"}):
        capital_costs = c.df.capital_cost*c.df[opt_name.get(c.name,"p") + "_nom_opt"]
        capital_costs_grouped = capital_costs.groupby(c.df.carrier).sum()

        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=["capital"])
        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(capital_costs_grouped.index.union(costs.index))

        costs.loc[capital_costs_grouped.index, label] = capital_costs_grouped

        if c.name == "Link":
            p = c.pnl.p0.multiply(n.snapshot_weightings.generators, axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
            p_all[p_all < 0.] = 0.
            p = p_all.sum()
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0).sum()

        #correct sequestration cost
        if c.name == "Store":
            items = c.df.index[(c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.)]
            c.df.loc[items, "marginal_cost"] = -20.

        marginal_costs = p*c.df.marginal_cost

        marginal_costs_grouped = marginal_costs.groupby(c.df.carrier).sum()

        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=["marginal"])
        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(marginal_costs_grouped.index.union(costs.index))

        costs.loc[marginal_costs_grouped.index,label] = marginal_costs_grouped

    # add back in all hydro
    #costs.loc[("storage_units", "capital", "hydro"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="hydro", "p_nom"].sum()
    #costs.loc[("storage_units", "capital", "PHS"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="PHS", "p_nom"].sum()
    #costs.loc[("generators", "capital", "ror"),label] = (0.02)*3e6*n.generators.loc[n.generators.group=="ror", "p_nom"].sum()

    return costs

def calculate_price_duration(n,label, price_duration):
    zone = snakemake.wildcards.zone
    area = snakemake.config['area']
    node = geoscope(n, zone, area)['node']
    price_duration = price_duration.reindex(columns=price_duration.columns.union(label))
    price_duration = price_duration.reindex(index=price_duration.index.union(n.buses_t.marginal_price.index))
    price_duration.loc[:,label] = n.buses_t.marginal_price[node].values
    return price_duration

def calculate_hydrogen_cost(n, label, h2_cost):
    if policy in ["ref"]: return h2_cost
    # revenue from selling and costs for buying
    marginal_price = n.buses_t.marginal_price
    # import
    p_import = n.links_t.p0.loc[:,n.links.carrier=="import"]
    p_export = n.links_t.p0.loc[:,n.links.carrier=="export"]
    # marginal price
    name = snakemake.config['ci']['name']
    zone = snakemake.wildcards.zone
    area = snakemake.config['area']
    node = geoscope(n, zone, area)['node']
    weight = n.snapshot_weightings.generators
    price = n.buses_t.marginal_price.loc[:, node].mul(weight)
    # import costs
    import_cost = p_import.mul(price.values, axis=0).sum()
    # export revenue
    export_cost = (-1)* p_export.mul(price.values, axis=0).sum()
    # offtake H2
    offtake = n.loads_t.p.loc[:,f"{name} H2"].mul(weight, axis=0).sum()
    # costs from the others
    try:
        costs = nodal_costs.loc[nodal_costs.index.get_level_values(2)==f"{name}",:][label].groupby(level=[0,3]).sum()
    except KeyError:
        print(label)
        return h2_cost
    if import_cost.empty:
        costs.loc[("links","import"),:] = 0
    else:
        costs.loc[("links","import"),:] = import_cost.values
    if export_cost.empty:
        costs.loc[("links","export"),:] = 0
    else:
        costs.loc[("links","export"),:] = export_cost.values
    costs.loc[("final", "H2 cost"),:] = (costs.drop("offtake H2", level=1, errors="ignore").sum()/abs(offtake))[0]
    costs.loc[("loads", "offtake H2"), label] = offtake
    h2_cost = h2_cost.reindex(index=h2_cost.index.union(costs.index))
    h2_cost = h2_cost.reindex(columns=h2_cost.columns.union(label))
    h2_cost.loc[costs.index, label] = costs

    return h2_cost

def calculate_emission_rate(n, label, emission_rate, attr_emissions):
    zone = snakemake.wildcards.zone
    area = snakemake.config['area']
    name = snakemake.config['ci']['name']
    country = geoscope(n, zone, area)['node']
    grid_clean_techs = snakemake.config['global']['grid_clean_techs']
    emitters = snakemake.config['global']['emitters']

    rest_system_buses = n.buses.index[~n.buses.index.str.contains(name) &
                                      (n.buses.location!=country)]
    country_buses = n.buses[n.buses.location==country].index


    clean_grid_generators =  n.generators.index[n.generators.bus.isin(rest_system_buses)
                                                & n.generators.carrier.isin(grid_clean_techs)]
    clean_grid_links = n.links.index[n.links.bus1.isin(rest_system_buses)
                                     & (n.links.carrier.isin(grid_clean_techs))]
    clean_grid_storage_units = n.storage_units.index[n.storage_units.bus.isin(rest_system_buses)
                                                     & n.storage_units.carrier.isin(grid_clean_techs)]
    dirty_grid_links = n.links.index[n.links.bus1.isin(rest_system_buses)
                                     & n.links.carrier.isin(emitters)]

    clean_country_generators = n.generators.index[n.generators.bus.isin(country_buses) & n.generators.carrier.isin(grid_clean_techs)]
    clean_country_links = n.links.index[n.links.bus1.isin(country_buses) & n.links.carrier.isin(grid_clean_techs)]
    clean_country_storage_units = n.storage_units.index[n.storage_units.bus.isin(country_buses) & n.storage_units.carrier.isin(grid_clean_techs)]
    dirty_country_links = n.links.index[n.links.bus1.isin(country_buses) & n.links.carrier.isin(emitters)]

    country_loads = n.loads.index[n.loads.bus.isin(country_buses)]

    clean_grid_gens = n.generators_t.p[clean_grid_generators].sum(axis=1)
    clean_grid_ls = (- n.links_t.p1[clean_grid_links].sum(axis=1))
    clean_grid_sus = n.storage_units_t.p[clean_grid_storage_units].sum(axis=1)
    clean_grid_resources = clean_grid_gens + clean_grid_ls + clean_grid_sus
    dirty_grid_resources = (- n.links_t.p1[dirty_grid_links].sum(axis=1))

    clean_country_gens = n.generators_t.p[clean_country_generators].sum(axis=1)
    clean_country_ls = (- n.links_t.p1[clean_country_links].sum(axis=1))
    clean_country_sus = n.storage_units_t.p[clean_country_storage_units].sum(axis=1)
    clean_country_resources = clean_country_gens + clean_country_ls + clean_country_sus
    dirty_country_resources = (- n.links_t.p1[dirty_country_links].sum(axis=1))

    line_imp_subsetA = n.lines_t.p1.loc[:,n.lines.bus0.str.contains(country)].sum(axis=1)
    line_imp_subsetB = n.lines_t.p0.loc[:,n.lines.bus1.str.contains(country)].sum(axis=1)
    line_imp_subsetA[line_imp_subsetA < 0] = 0.
    line_imp_subsetB[line_imp_subsetB < 0] = 0.

    links_imp_subsetA = n.links_t.p1.loc[:,n.links.bus0.str.contains(country) &
                        (n.links.carrier == "DC") & ~(n.links.index.str.contains(name))].sum(axis=1)
    links_imp_subsetB = n.links_t.p0.loc[:,n.links.bus1.str.contains(country) &
                        (n.links.carrier == "DC") & ~(n.links.index.str.contains(name))].sum(axis=1)
    links_imp_subsetA[links_imp_subsetA < 0] = 0.
    links_imp_subsetB[links_imp_subsetB < 0] = 0.

    country_import =   line_imp_subsetA + line_imp_subsetB + links_imp_subsetA + links_imp_subsetB

    grid_hourly_emissions = n.links_t.p0[dirty_grid_links].multiply(n.links.efficiency2[dirty_grid_links],axis=1).sum(axis=1)

    grid_emission_rate =  grid_hourly_emissions / (clean_grid_resources + dirty_grid_resources)

    grid_hourly_emissions = n.links_t.p0[dirty_grid_links].multiply(n.links.efficiency2[dirty_grid_links],axis=1).sum(axis=1)
    grid_hourly_emissions_by_car = n.links_t.p0[dirty_grid_links].multiply(n.links.efficiency2[dirty_grid_links],axis=1).groupby(n.links.carrier,axis=1).sum()

    grid_emission_rate =  grid_hourly_emissions / (clean_grid_resources + dirty_grid_resources)
    grid_emission_rate_by_car = grid_hourly_emissions_by_car.div(clean_grid_resources + dirty_grid_resources, axis=0)

    country_hourly_emissions = n.links_t.p0[dirty_country_links].multiply(n.links.efficiency2[dirty_country_links],axis=1).sum(axis=1)
    country_hourly_emissions_by_car = n.links_t.p0[dirty_country_links].multiply(n.links.efficiency2[dirty_country_links],axis=1).groupby(n.links.carrier, axis=1).sum()

    grid_supply_emission_rate = (country_hourly_emissions + country_import*grid_emission_rate) / \
                                (clean_country_resources + dirty_country_resources + country_import)
    grid_supply_emission_rate_by_car = ((country_hourly_emissions_by_car
                                        + grid_emission_rate_by_car
                                        .mul(country_import, axis=0))
                                        .div(clean_country_resources
                                             + dirty_country_resources
                                             + country_import, axis=0))
    grid_supply_emission_rate_withoutimports = country_hourly_emissions / \
                                (clean_country_resources + dirty_country_resources)
    grid_supply_emission_rate_by_car_withoutimports = (country_hourly_emissions_by_car
                                        .div(clean_country_resources
                                             + dirty_country_resources, axis=0))

    ci_emissions_t = n.links_t.p0[f"{name} import"]*grid_supply_emission_rate
    ci_emissions_t_ni = n.links_t.p0[f"{name} import"]*grid_supply_emission_rate_withoutimports
    ci_emissions_t_by_carrier = grid_supply_emission_rate_by_car.mul(n.links_t.p0[f"{name} import"], axis=0)
    ci_emissions_t_by_carrier_ni = grid_supply_emission_rate_by_car_withoutimports.mul(n.links_t.p0[f"{name} import"], axis=0)

    carbon_intensity_h2 = ci_emissions_t.sum()/abs(n.loads_t.p.loc[:,f"{name} H2"].sum()) * LHV_H2
    carbon_intesity_h2_by_car = ci_emissions_t_by_carrier.sum().div(n.loads_t.p.loc[:,f"{name} H2"].sum())* LHV_H2
    carbon_intesity_h2_by_car = pd.concat([carbon_intesity_h2_by_car], keys=["with imports"])
    carbon_intensity_h2_ni = ci_emissions_t_ni.sum()/abs(n.loads_t.p.loc[:,f"{name} H2"].sum()) * LHV_H2
    carbon_intesity_h2_by_car_ni = ci_emissions_t_by_carrier_ni.sum().div(n.loads_t.p.loc[:,f"{name} H2"].sum())* LHV_H2
    carbon_intesity_h2_by_car_ni = pd.concat([carbon_intesity_h2_by_car_ni], keys=["no imports"])

    emission_rate = emission_rate.reindex(columns = emission_rate.columns.union(label))
    emission_rate.loc["carbon_intensity_H2", label] = carbon_intensity_h2
    emission_rate.loc["carbon_intensity_H2_ni", label] = carbon_intensity_h2_ni
    emission_rate.loc["ci_emissions", label] = ci_emissions_t.sum()

    new_index = carbon_intesity_h2_by_car.index.union(carbon_intesity_h2_by_car_ni.index)
    attr_emissions = attr_emissions.reindex(columns = attr_emissions.columns.union(label))
    attr_emissions = attr_emissions.reindex(index= attr_emissions.index.union(new_index))
    attr_emissions.loc[carbon_intesity_h2_by_car.index, label] = carbon_intesity_h2_by_car.values
    attr_emissions.loc[carbon_intesity_h2_by_car_ni.index, label] = carbon_intesity_h2_by_car_ni.values

    return emission_rate, attr_emissions



#%%
final = pd.DataFrame()
cf = pd.DataFrame()
emissions = pd.DataFrame()
supply_energy = pd.DataFrame()
nodal_capacities = pd.DataFrame()
weighted_prices = pd.DataFrame()
curtailment = pd.DataFrame()
nodal_costs = pd.DataFrame()
costs = pd.DataFrame()
h2_cost = pd.DataFrame()
emission_rate = pd.DataFrame()
nodal_supply_energy = pd.DataFrame()
h2_gen_mix = pd.DataFrame()
attr_emissions = pd.DataFrame()
price_duration = pd.DataFrame()
name = snakemake.config["ci"]["name"]

# network
n = pypsa.Network(snakemake.input.network,
                  override_component_attrs=override_component_attrs())
assign_locations(n, name)
weightings = n.snapshot_weightings.generators

# wildcards
policy = snakemake.wildcards.policy
country = snakemake.wildcards.zone
year = snakemake.wildcards.year
ct_target = snakemake.config[f"res_target_{year}"][country]
res_share = snakemake.wildcards.res_share
storage_type = snakemake.wildcards.storage
volume = snakemake.wildcards.offtake_volume
palette = snakemake.wildcards.palette

ct_target = snakemake.config[f"res_target_{year}"][country]
if res_share=="p0":
    res_share = ct_target
else:
    res_share = round(float(res_share.replace("m","-").replace("p",".")), ndigits=2)


cols = pd.MultiIndex.from_product([[policy], [res_share], [volume], [storage_type]],
                                  names=["policy", "res_share", "volume", "storage_type"])

# capacity factor
p_nom_opt = n.links.p_nom_opt
df = (n.links_t.p0/n.links.p_nom_opt)
df_s = (n.stores_t.e/n.stores.e_nom_opt)
df = df.loc[:, p_nom_opt[p_nom_opt>10].index]
df.dropna(axis=1, inplace=True)
cols = pd.MultiIndex.from_product([[policy], [res_share], [volume],
                                   [storage_type], df.columns],
                                  names=["policy", "res_share",
                                         "volume", "storage_type", "node"])
df.columns = cols
cols = pd.MultiIndex.from_product([[policy], [res_share], [volume],
                                   [storage_type], df_s.columns],
                                  names=["policy", "res_share",
                                         "volume", "storage_type", "node"])
df_s.columns = cols
cf = pd.concat([cf, df, df_s], axis=1)

cols = pd.MultiIndex.from_product([ [policy], [res_share], [volume], [storage_type]],
                                  names=["policy", "res_share", "volume", "storage_type"])
# co2 emissions
co2_emission = n.stores_t.e["co2 atmosphere"].iloc[-1]
co2_emission = pd.DataFrame([co2_emission], index=cols)
emissions = pd.concat([emissions, co2_emission])

# supply energy
supply_energy = calculate_supply_energy(n, cols, supply_energy)
# nodal supply energy
nodal_supply_energy = calculate_nodal_supply_energy(n, cols, nodal_supply_energy)
# nodal capacities
nodal_capacities = calculate_nodal_capacities(n, cols, nodal_capacities)

weighted_prices = calculate_weighted_prices(n, cols, weighted_prices)

curtailment = calculate_curtailment(n,cols, curtailment)

# nodal capacities
nodal_costs = calculate_nodal_costs(n, cols, nodal_costs)

# costs
costs = calculate_costs(n, cols, costs)

# h2 costs
h2_cost = calculate_hydrogen_cost(n, cols, h2_cost)

# carbon intensity H2
if "import" in n.links.carrier.unique():
    emission_rate, attr_emissions = calculate_emission_rate(n, cols, emission_rate, attr_emissions)

if policy!="ref":
    h2_gen_mix = calculate_h2_generationmix(n, cols, h2_gen_mix)
# calculate price duration in local zone
price_duration = calculate_price_duration(n,cols, price_duration)
#%%
cf = cf.loc[:,((cf.columns.get_level_values(4).str.contains(f"{name} H2 Electrolysis"))|
            (cf.columns.get_level_values(4).str.contains(f"{name} H2 Store")))]


cf.to_csv(snakemake.output.csvs_cf)
supply_energy.to_csv(snakemake.output.csvs_supply_energy)
nodal_supply_energy.to_csv(snakemake.output.csvs_nodal_supply_energy)
nodal_capacities.to_csv(snakemake.output.csvs_nodal_capacities)
weighted_prices.to_csv(snakemake.output.csvs_weighted_prices)
curtailment.to_csv(snakemake.output.csvs_curtailment)
costs.to_csv(snakemake.output.csvs_costs)
nodal_costs.to_csv(snakemake.output.csvs_nodal_costs)
h2_cost.to_csv(snakemake.output.csvs_h2_costs)
emission_rate.to_csv(snakemake.output.csvs_emission_rate)
h2_gen_mix.to_csv(snakemake.output.csvs_h2_gen_mix)
attr_emissions.to_csv(snakemake.output.csvs_attr_emissions)
price_duration.to_csv(snakemake.output.csvs_price_duration)
emissions.to_csv(snakemake.output.csvs_emissions)
