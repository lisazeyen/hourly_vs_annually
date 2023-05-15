#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:32:11 2022

@author: lisa
"""
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        import os
        os.chdir("/home/lisa/Documents/hourly_vs_annually/scripts")
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('plot_offtake', palette='p1',
                                   zone='DE', year='2025',  participation='10',
                                   policy="ref")
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

snakemake.config['tech_colors']["solar local"] = "#ffdd78"
snakemake.config['tech_colors']["onwind local"] = "#bae3f9"
for k,v in rename_techs.items():
    snakemake.config['tech_colors'][v] = snakemake.config['tech_colors'][k]

name = snakemake.config['ci']['name']

year = snakemake.wildcards.year
country = snakemake.wildcards.zone
res_share = str(snakemake.config[f"res_target_{year}"][country])
#%%

def plot_series(network,label, carrier="AC"):

    n = network.copy()
    # assign_location(n)
    # assign_carriers(n)

    buses = n.buses.index[n.buses.carrier.str.contains(carrier)]
    buses = [f"{name}"]

    supply = pd.DataFrame(index=n.snapshots)
    for c in n.iterate_components(n.branch_components):
        n_port = 4 if c.name=='Link' else 2
        for i in range(n_port):
            supply = pd.concat((supply,
                                (-1) * c.pnl["p" + str(i)].loc[:,
                                                               c.df.index[c.df["bus" + str(i)].isin(buses)]].groupby(c.df.carrier,
                                                                                                                     axis=1).sum()),
                               axis=1)

    for c in n.iterate_components(n.one_port_components):
        comps = c.df.index[c.df.bus.isin(buses)]
        supply = pd.concat((supply, ((c.pnl["p"].loc[:, comps]).multiply(
            c.df.loc[comps, "sign"])).groupby(c.df.carrier, axis=1).sum()), axis=1)

    supply = supply.groupby(supply.columns, axis=1).sum()

    both = supply.columns[(supply < 0.).any() & (supply > 0.).any()]

    positive_supply = supply[both]
    negative_supply = supply[both]

    positive_supply[positive_supply < 0.] = 0.
    negative_supply[negative_supply > 0.] = 0.

    supply[both] = positive_supply

    suffix = " charging"

    negative_supply.columns = negative_supply.columns + suffix

    supply = pd.concat((supply, negative_supply), axis=1)

    # 14-21.2 for flaute
    # 19-26.1 for flaute


    threshold = 1

    to_drop = supply.columns[(abs(supply) < threshold).all()]

    if len(to_drop) != 0:
        print("dropping", to_drop)
        supply.drop(columns=to_drop, inplace=True)

    if supply.empty: return

    supply.index.name = None

    supply = supply / 1e3

    supply.rename(columns={"electricity": "electric demand",
                           "heat": "heat demand"},
                  inplace=True)
    supply.columns = supply.columns.str.replace("residential ", "")
    supply.columns = supply.columns.str.replace("services ", "")
    supply.columns = supply.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index([ "solar PV", "onshore wind", "batttery discharger",
                                "purchase",
                             "electrolysis",
                             "battery charger", "sale"])

    supply.rename(columns=rename_techs, inplace=True)
    new_columns = (preferred_order.intersection(supply.columns)
                   .append(supply.columns.difference(preferred_order)))

    year = snakemake.wildcards.year
    zone = snakemake.wildcards.zone
    supply =  supply.groupby(supply.columns, axis=1).sum()
    snakemake.config["tech_colors"]["PHS charging"] = snakemake.config["tech_colors"]["PHS"]
    snakemake.config["tech_colors"]["electric demand"] = snakemake.config["tech_colors"]["AC"]
    snakemake.config["tech_colors"]["offtake H2"] = "#FFC0CB"
    supply.rename(index=lambda x: x.replace(year = int(year)),
                  inplace=True)


    starts = [f"{year}-03-01", f"{year}-12-21"]
    stops = [f"{year}-03-08", f"{year}-12-28"]

    for i, start in enumerate(starts):
        stop = stops[i]
        fig, ax = plt.subplots()
        fig.set_size_inches((8, 5))

        (supply.loc[start:stop, new_columns]
         .plot(ax=ax, kind="area", stacked=True, linewidth=0.,
               color=[snakemake.config['tech_colors'][i.replace(suffix, "")]
                      for i in new_columns]))

        handles, labels = ax.get_legend_handles_labels()

        handles.reverse()
        labels.reverse()

        new_handles = []
        new_labels = []

        for i, item in enumerate(labels):
            if "charging" not in item:
                new_handles.append(handles[i])
                new_labels.append(labels[i])

        ax.legend(new_handles, new_labels, ncol=3, loc="upper left", frameon=False)
        ax.set_xlim([start, stop])
        # ax.set_ylim([-1300, 1900])
        ax.grid(True)
        ax.set_ylabel("Power [GW]")
        fig.tight_layout()

        run = snakemake.config["run"]
        fig.savefig(snakemake.output.cf_plot.split("graphs/")[0] +
                                                   "series-{}-{}-{}-{}-{}-{}-{}-{}.pdf".format(
            zone,  label.to_list()[0][0],
            label.to_list()[0][1],label.to_list()[0][2],
            label.to_list()[0][3],  start, stop, year),
            transparent=True)


def plot_nodal_balances(nodal_supply_energy):
    co2_carriers = ["co2", "co2 stored", "process emissions"]


    balances = {i.replace(" ","_"): [i] for i in nodal_supply_energy.index.levels[0]}
    balances["energy"] = [i for i in nodal_supply_energy.index.levels[0] if i not in co2_carriers]

    zone = snakemake.wildcards.zone

    k = "AC"
    v = ["AC"]

    df = nodal_supply_energy.loc[v].xs(zone, level=2)
    df = df.groupby(df.index.get_level_values(2)).sum()

    #convert MWh to TWh
    df = df / 1e6

    #remove trailing link ports
    df.index = [i[:-1] if ((i not in ["co2", "NH3", "H2"]) and (i[-1:] in ["0","1","2","3"])) else i for i in df.index]

    # df = df.groupby(df.index.map(rename_techs)).sum()
    df = df.groupby(df.index).sum()

    df = df.droplevel([0,2], axis=1)

    df = df["ref"].iloc[:,:1]

    to_drop = df.index[df.abs().max(axis=1) < 1] # snakemake.config['plotting']['energy_threshold']/10]

    print("dropping")

    print(df.loc[to_drop])

    df = df.drop(to_drop)


    df.drop(["AC", "DC"], errors="ignore",inplace=True)
    df.rename(index=rename_techs,inplace=True)
    df = df.groupby(df.index).sum()
    df.columns = [f"{snakemake.wildcards.zone}-{snakemake.wildcards.year}"]

    a = df[df>0].dropna()


    wished_tech_order = pd.Index(["electricity", "hydro", "offshore wind",
                         "onshore wind", "solar PV", "biomass CHP",
                         'CCGT', 'coal', "lignite", "oil"])
    order = wished_tech_order.intersection(a.index).union(a.index.difference(wished_tech_order))
    a = a.reindex(order)


    fig, ax = plt.subplots()
    pd.DataFrame(a).T.plot(kind="bar", ax=ax, stacked=True, grid=True,
                         color=[snakemake.config['tech_colors'][i] for i in a.index])
    plt.ylabel("generation \n TWh/a")
    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"energy-{zone}-{k}-noimports.pdf",
                bbox_inches='tight')


def plot_balances(balances_df):

    co2_carriers = ["co2", "co2 stored", "process emissions"]


    balances = {i.replace(" ","_"): [i] for i in balances_df.index.levels[0]}
    balances["energy"] = [i for i in balances_df.index.levels[0] if i not in co2_carriers]

    for k, v in balances.items():

        df = balances_df.loc[v]
        df = df.groupby(df.index.get_level_values(2)).sum()

        #convert MWh to TWh
        df = df / 1e6

        #remove trailing link ports
        df.index = [i[:-1] if ((i not in ["co2", "NH3", "H2"]) and (i[-1:] in ["0","1","2","3"])) else i for i in df.index]

        # df = df.groupby(df.index.map(rename_techs)).sum()
        df = df.groupby(df.index).sum()

        df = df.droplevel([1,2], axis=1)


        to_drop = df.index[df.abs().max(axis=1) < 1] # snakemake.config['plotting']['energy_threshold']/10]


        df = df.drop(to_drop)

        # df = df.droplevel(0, axis=1)

        print(df.sum())

        if df.empty:
            continue

        #for price in df.columns.levels[1]:
        for policy in ["grd", "ref", "res"]:
            for volume in df.columns.levels[2]:
                for storage_type in ["nostore", "flexibledemand"]: #df.columns.levels[3]:
                    fig, ax = plt.subplots(figsize=(12,8))

                    balance = df.xs((volume,storage_type),level=[2,3], axis=1)
                    to_drop = ["offgrid", "cfe", "grd", "ref"]
                    if policy=="res": to_drop.append("res")
                    balance.sub(balance[policy], axis=0, level=1).drop(to_drop, errors="ignore",axis=1).T.plot(kind="bar",ax=ax,stacked=True,
                                                            color=[snakemake.config['tech_colors'][i] for i in df.index],
                                                            grid=True,
                                                            title= f"Difference policy - {policy}: volume {volume} storage type {storage_type}")

                    handles,labels = ax.get_legend_handles_labels()

                    handles.reverse()
                    labels.reverse()

                    if v[0] in co2_carriers:
                        ax.set_ylabel("CO2 [MtCO2/a]")
                    else:
                        ax.set_ylabel("Energy [TWh/a]")

                    ax.set_xlabel("")

                    ax.grid(axis="x")

                    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)


                    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"balances_difference_{policy}" + k + f"_{volume}volume_{storage_type}.pdf", bbox_inches='tight')

                    # generation diff ############
                    generation = balance[(balance>0)]

                    fig, ax = plt.subplots(figsize=(12,8))
                    generation.sub(generation[policy], axis=0, level=1).drop(to_drop, errors="ignore",axis=1).T.plot(kind="bar",ax=ax,stacked=True,
                                                            color=[snakemake.config['tech_colors'][i] for i in generation.index],
                                                            grid=True,
                                                            title= f"Difference policy - {policy}: volume {volume} storage type {storage_type}")
                    handles,labels = ax.get_legend_handles_labels()

                    handles.reverse()
                    labels.reverse()

                    if v[0] in co2_carriers:
                        ax.set_ylabel("CO2 [MtCO2/a]")
                    else:
                        ax.set_ylabel("Energy [TWh/a]")

                    ax.set_xlabel("")

                    ax.grid(axis="x")

                    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)


                    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"balances_difference_generation_{policy}" + k + f"_{volume}volume_{storage_type}.pdf", bbox_inches='tight')


                    # demand diff ############
                    generation = (-1)*balance[(balance<0)]

                    fig, ax = plt.subplots(figsize=(12,8))
                    generation.sub(generation[policy], axis=0, level=1).drop(to_drop, errors="ignore",axis=1).T.plot(kind="bar",ax=ax,stacked=True,
                                                            color=[snakemake.config['tech_colors'][i] for i in generation.index],
                                                            grid=True,
                                                            title= f"Difference policy - {policy}: volume {volume} storage type {storage_type}")
                    handles,labels = ax.get_legend_handles_labels()

                    handles.reverse()
                    labels.reverse()

                    if v[0] in co2_carriers:
                        ax.set_ylabel("CO2 [MtCO2/a]")
                    else:
                        ax.set_ylabel("Energy [TWh/a]")

                    ax.set_xlabel("")

                    ax.grid(axis="x")

                    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)


                    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"balances_difference_demand_{policy}" + k + f"_{volume}volume_{storage_type}.pdf", bbox_inches='tight')

        for policy in df.columns.levels[0]:
            for volume in df.columns.levels[2]:
                for storage_type in df.columns.levels[3]:
                    fig, ax = plt.subplots(figsize=(12,8))

                    balance = df.xs((policy,volume,storage_type),level=[0,2,3], axis=1)

                    balance.T.plot(kind="bar",ax=ax,stacked=True,
                                                           color=[snakemake.config['tech_colors'][i] for i in df.index],
                                                           grid=True,
                                                           title= f"policy {policy} volume {volume} storage type {storage_type}")


                    handles,labels = ax.get_legend_handles_labels()

                    handles.reverse()
                    labels.reverse()

                    if v[0] in co2_carriers:
                        ax.set_ylabel("CO2 [MtCO2/a]")
                    else:
                        ax.set_ylabel("Energy [TWh/a]")

                    ax.set_xlabel("")

                    ax.grid(axis="x")

                    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)


                    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + "balances_" + k + f"_{policy}policy_{volume}volume_{storage_type}.pdf", bbox_inches='tight')


def plot_duration_curve(cf, tech, wished_policies, wished_order, volume, name=""):
    cf_elec = cf.xs((res_share, tech), level=[1,4],axis=1)[wished_policies].xs(volume,level=1, axis=1)
    cf_elec = cf_elec.reindex(wished_order, level=1, axis=1)
    cf_elec.rename(rename_scenarios,
                   level=0,inplace=True, axis=1)
    cf_elec = cf_elec.apply(lambda x: x.sort_values(ascending=False).reset_index(drop=True), axis=0)
    cf_elec.rename(index=lambda x: 3*x, inplace=True)

    fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies),
                           sharey=True, sharex=True,
                           figsize=(10,1.5))
    nice_names = [rename_scenarios[scen] if scen in rename_scenarios.keys() else scen for scen in wished_policies]
    for i, policy in enumerate(nice_names):
        cf_elec[policy].plot(grid=True, ax=ax[i], title=policy, lw=2, legend=False)
        ax[i].set_xlabel("")
        ax[i].set_xlabel("")
        ax[i].grid(alpha=0.3)
        ax[i].set_axisbelow(True)

    ax[0].set_ylabel("capacity factor")
    ax[1].set_xlabel("hours")
    ax[0].set_xlim([0,8760])
    ax[0].set_ylim(bottom=0)
    plt.legend(bbox_to_anchor=(1,1))
    fig.savefig(snakemake.output.cf_plot.split("cf_elec")[0] + f"duration_curve_{volume}{name}.pdf",
                bbox_inches="tight")



def plot_cf(df, wished_policies, wished_order, volume, name=""):
    df = df[~df.index.duplicated()]
    df = df.xs(res_share, level=1)
    cf_elec = df.loc[wished_policies].xs(volume, level=1)
    cf_elec = cf_elec.reindex(wished_order, level=1)
    cf_elec.rename(index=rename_scenarios,
                   level=0,inplace=True)
    fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies), sharey=True,
                           figsize=(10, 1.5))
    nice_names = [rename_scenarios[scen] if scen in rename_scenarios.keys() else scen for scen in wished_policies]
    for i, policy in enumerate(nice_names):
        cf_elec.loc[policy].plot(kind="bar", grid=True, ax=ax[i], title=policy,
                                 width=0.65)
        ax[i].set_xlabel("")
        ax[i].grid(alpha=0.3)
        ax[i].set_axisbelow(True)
    ax[0].set_ylabel("capacity factor")
    fig.savefig(snakemake.output.cf_plot.split(".pdf")[0] + f"{volume}{name}.pdf",
                bbox_inches="tight")

    fig.savefig(snakemake.output.cf_plot,
                bbox_inches="tight")


def plot_consequential_emissions(emissions, supply_energy, wished_policies,
                                 wished_order, volume, name=""):
    compare_p = "ref" if snakemake.config["solving_option"]!="together" else "exl1p0"
    # consequential emissions
    emissions = emissions[~emissions.index.duplicated()]
    emissions_v = emissions.loc[wished_policies+[compare_p]].xs((float(res_share), float(volume)), level=[1,2])
    emissions_v = emissions_v.reindex(wished_order, level=1)
    emissions_v.rename(index=rename_scenarios,
                       level=0,inplace=True)
    emissions_s = (supply_energy.loc["co2"][wished_policies+[compare_p]]
                   .xs((res_share, str(volume)),level=[1,2], axis=1))
    emissions_s = emissions_s.loc[:, ~emissions_s.columns.duplicated()]
    emissions_s = emissions_s.reindex(wished_order, level=1, axis=1)
    emissions_s = emissions_s[emissions_s>0].dropna(axis=0).rename(index=lambda x:x.replace("2",""))
    emissions_s.rename(rename_scenarios,level=0,inplace=True, axis=1)
    emissions_s = emissions_s.droplevel(0)

    nice_names = [rename_scenarios[scen] if scen in rename_scenarios.keys()
                  else scen for scen in wished_policies]
    compare_p = [rename_scenarios[scen] if scen in rename_scenarios.keys()
                  else scen for scen in [compare_p]]
    emissions_v = emissions_v.groupby(level=[0,1]).first()
    emissions_v = emissions_v.reindex(wished_order, level=1)
    fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies), sharey=True,figsize=(10,1.5))
    for i, policy in enumerate(nice_names):
        # annually produced H2 in [t_H2/a]
        produced_H2 = float(volume)*8760 / LHV_H2
        em_p = emissions_v.loc[policy].sub(emissions_v.loc[compare_p].mean())/ produced_H2

        em_p.iloc[:,0].plot(kind="bar", grid=True, ax=ax[i], title=policy,
                            width=0.65)
        ax[i].set_xlabel("")
        ax[i].grid(alpha=0.3)
        ax[i].set_axisbelow(True)
    ax[0].set_ylabel("consequential emissions \n [kg$_{CO_2}$/kg$_{H_2}$]")
    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0]+ f"consequential_emissions_{volume}{name}.pdf",
                bbox_inches="tight")

    y_min = 0
    y_max = 0
    emissions_s = emissions_s.groupby(level=[0,1], axis=1).first()
    emissions_s = emissions_s.reindex(wished_order, level=1, axis=1)
    fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies), sharey=True,figsize=(10,1.5))
    for i, policy in enumerate(nice_names):
        # annually produced H2 in [t_H2/a]
        produced_H2 = float(volume)*8760 / LHV_H2
        em_p = emissions_s[policy].sub(emissions_s[compare_p[0]])/ produced_H2

        em_p.sum().rename("net total").plot(ax=ax[i], lw=0, marker="_", color="black",
                                            markersize=15, markeredgewidth=2)
        # em_p.sum().rename("net total").plot(ax=ax[i], kind="bar", color="black",
        #                                     width=0.25, position=-1)
        em_p.T.plot(kind="bar", stacked=True, grid=True, ax=ax[i], title=policy,
                            width=0.65, legend=False,
                            color=[snakemake.config['tech_colors'][i] for i in em_p.index])
        # import pyam

        # from pyam.plotting import add_net_values_to_bar_plot
        # add_net_values_to_bar_plot(ax[i], color='k')
        # # fig.subplots_adjust(right=0.55)

        ax[i].set_xlabel("")
        ax[i].grid(alpha=0.3)
        ax[i].set_axisbelow(True)
        if em_p[em_p<0].sum().min()<y_min: y_min = em_p[em_p<0].sum().min()
        if em_p[em_p>0].sum().max()>y_max: y_max = em_p[em_p>0].sum().max()
    ax[0].set_ylim([1.1*y_min, 1.1*y_max])
    ax[0].set_ylabel("consequential emissions \n [kg$_{CO_2}$/kg$_{H_2}$]")
    plt.legend(fontsize=9, bbox_to_anchor=(1,1))
    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0]+ f"consequential_emissions_by_carrier_{volume}{name}.pdf",
                bbox_inches="tight")



def plot_attributional_emissions(attr_emissions, wished_policies, wished_order, volume, name=""):
    for consider_import in ['no imports', 'with imports']:
        attr_emissions = attr_emissions[~attr_emissions.index.duplicated()]
        em_r = attr_emissions.loc[consider_import].xs((res_share, volume), level=[1,2], axis=1)
        em_r = em_r.loc[:, ~em_r.columns.duplicated()]
        em_r = em_r.stack().reindex(wished_policies,axis=1).fillna(0).unstack()
        em_r = em_r.reindex(wished_order, axis=1, level=1)
        em_r.rename(columns=rename_scenarios,
                           level=0,inplace=True)
        nice_names = [rename_scenarios[scen] if scen in rename_scenarios.keys()
                      else scen for scen in wished_policies]
        width = len(wished_policies)*2
        figsize=(width,1.5)
        fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies), sharey=True,figsize=figsize,
                               )
        for i, policy in enumerate(nice_names):
            em_r[policy].T.plot(kind="bar", stacked=True, grid=True, ax=ax[i],
                                title=policy,
                                color=[snakemake.config['tech_colors'][i] for i in em_r.index],
                                legend=False,
                                 width=0.65)
            ax[i].grid(alpha=0.3)
            ax[i].set_axisbelow(True)
            ax[i].set_xlabel("")
            ax[i].axhline(y=1, linestyle="--", color="black")
            ax[i].axhline(y=3, linestyle="--", color="black")
            ax[i].axhline(y=10, linestyle="--", color="black")
        ax[0].set_ylabel("attributional emissions\n [kg$_{CO_2}$/kg$_{H_2}$]")
        ax[len(wished_policies)-1].text(x= 0.93*len(wished_order), y=0., s='carbon intensity of\nblue hydrogen')
        ax[len(wished_policies)-1].text(x= 0.93*len(wished_order), y=8, s='carbon intensity of\ngrey hydrogen')
        ax[len(wished_policies)-1].text(x=0.93*len(wished_order), y=2.8, s='EU threshold for \nlow-carbon hydrogen')
        if consider_import == "with imports":
            suffix = ""
        else:
            suffix = "_noimports"
        plt.legend(ncol=2, bbox_to_anchor=(1,1), loc="upper left", fontsize=10) if len(wished_policies)==2  else  plt.legend(ncol=1,fontsize=10, bbox_to_anchor=(2.9, 1.))
        # plt.legend(ncol=2) # bbox_to_anchor=(1,0.85), loc="upper left")
        plt.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"attributional_emissions_{volume}{name}{suffix}.pdf",
                    bbox_inches='tight')


def plot_cost_breakdown(h2_cost, wished_policies, wished_order, volume, name=""):
    h2_cost = h2_cost[~h2_cost.index.duplicated()]
    costb = h2_cost.xs((res_share, volume), level=[1,2], axis=1).droplevel(0)
    costb = costb.loc[:, ~costb.columns.duplicated()]
    costb = costb.stack().reindex(wished_policies, axis=1).fillna(0).unstack()
    costb = costb.reindex(wished_order, level=1, axis=1)
    costb.rename(columns=rename_scenarios,
                       level=0,inplace=True)
    costb.loc["battery"] = costb[costb.index.str.contains("battery")].sum()
    costb.drop(["battery charger", "battery discharger"], inplace=True)
    singlecost = costb.drop(["H2 cost", "offtake H2"]).div(float(volume)*8760)
    # convert EUR/MWh_H2 in Eur/kg_H2
    singlecost = singlecost / 1e3 * LHV_H2
    to_drop = singlecost.loc[abs(singlecost.sum(axis=1))<1].index
    singlecost.drop(to_drop, inplace=True)
    nice_names = [rename_scenarios[scen] if scen in rename_scenarios.keys()
                  else scen for scen in wished_policies]


    singlecost.rename(index=rename_techs, inplace=True)
    wished_tech_order = ['sale', 'H2 store', 'battery',
                         'electrolysis', 'onshore wind', 'solar PV', 'load shedding', 'purchase',]
    singlecost = singlecost.reindex(index=wished_tech_order)


    fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies), sharey=True,figsize=(10,1.5))
    for i, policy in enumerate(nice_names):
        singlecost.sum().loc[policy].rename("net total").plot(ax=ax[i], marker="_",
                                                              lw=0, color="black",
                                                              markersize=15, markeredgewidth=2)
        singlecost.T.loc[policy].plot(kind="bar", stacked=True, ax=ax[i], title=policy,
                 color=[snakemake.config['tech_colors'][i] for i in singlecost.index],
                 grid=True, legend=False,  width=0.65)
        ax[i].grid(alpha=0.3)
        ax[i].set_axisbelow(True)
    not_grd = singlecost.columns!=(    'grid',        'nostore')
    ax[0].set_ylim([singlecost[singlecost<0].sum().min()*1.1, singlecost[singlecost>0].sum().loc[not_grd].max()*1.1])
    ax[0].set_ylabel("cost \n [Euro/kg$_{H_2}$]")
    plt.legend(bbox_to_anchor=(1,1))
    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"costbreakdown_{volume}{name}.pdf",
                bbox_inches='tight')



def plot_shadow_prices(weighted_prices, wished_policies, wished_order, volume,
                       name="", carrier="AC"):
    weighted_prices = weighted_prices[~weighted_prices.index.duplicated()]
    w_price = weighted_prices.xs((res_share, volume, "rest"), level=[1,2, 4], axis=1)[wished_policies].loc[carrier]
    w_price.rename(rename_scenarios, level=0, inplace=True)
    nice_names = [rename_scenarios[scen] if scen in rename_scenarios.keys()
                  else scen for scen in wished_policies]
    w_price = w_price.loc[~w_price.index.duplicated()]
    not_grd = w_price.index!=("grid", "nostore")

    fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies),
                           sharey=True,figsize=(10,1.5))
    for i, policy in enumerate(nice_names):
        w_price.loc[policy].reindex(wished_order).plot(kind="bar", ax=ax[i],
                                                 title=policy, grid=True, width=0.65)
        ax[i].grid(alpha=0.3)
        ax[i].set_axisbelow(True)

    ax[0].set_ylabel("price \n [Euro/MWh]")
    ax[0].set_ylim([0, w_price.loc[not_grd].max()*1.1])
    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"shadowprices_{volume}{name}_{carrier}_croped.pdf",
                bbox_inches='tight')

    no_hourly = [x for x in nice_names if "hourly" not in x]

    if len(no_hourly)!=0:
        fig, ax = plt.subplots(nrows=1, ncols=len(no_hourly),
                               sharey=True,figsize=(10,1.5))
        for i, policy in enumerate(no_hourly):
            w_price.loc[policy].reindex(wished_order).plot(kind="bar", ax=ax[i],
                                                     title=policy, grid=True, width=0.65)
            ax[i].grid(alpha=0.3)
            ax[i].set_axisbelow(True)

        ax[0].set_ylabel("price \n [Euro/MWh]")
        ax[0].set_ylim([0, w_price.loc[no_hourly].max()*1.1])
        fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"shadowprices_{volume}{name}_{carrier}_croped_without_hourly.pdf",
                    bbox_inches='tight')

def plot_h2genmix(h2_gen_mix, wished_policies, wished_order, volume,
                       name=""):
    h2_gen_mix = h2_gen_mix[~h2_gen_mix.index.duplicated()]
    gen_mix = h2_gen_mix.xs((res_share, volume), level=[1,2], axis=1)[wished_policies]
    gen_mix.rename(rename_scenarios, level=0, axis=1, inplace=True)
    nice_names = [rename_scenarios[scen] if scen in rename_scenarios.keys()
                  else scen for scen in wished_policies]

    gen_mix = gen_mix[nice_names][gen_mix[nice_names].sum(axis=1)>1]
    gen_mix.rename(index=lambda x: x.replace("urban central solid ", ""), inplace=True)
    gen_mix.loc["offwind",:] = gen_mix[gen_mix.index.str.contains("offwind")].sum()
    gen_mix.drop(["offwind-ac", "offwind-dc"],inplace=True, errors="ignore")

    carrier_order = ["solar local", "onwind local", "solar", "onwind",  "offwind",
                     "ror", "biomass CHP", "CCGT", "OCGT", "coal", "lignite", "oil"]
    gen_mix = gen_mix.reindex(carrier_order)
    gen_mix = gen_mix.rename(index=rename_techs)
    gen_mix = gen_mix.loc[:, ~gen_mix.columns.duplicated()]
    fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies),
                           sharey=True,figsize=(10,1.5))
    for i, policy in enumerate(nice_names):
        (gen_mix[policy]/1e6).reindex(columns=wished_order).T.plot(kind="bar", stacked=True,
                                                             ax=ax[i], width=0.65,
                                                             color=[snakemake.config['tech_colors'][i] for i in gen_mix.index],
                                                 title=policy, grid=True, legend=False)
        ax[i].grid(alpha=0.3)
        ax[i].set_axisbelow(True)

    plt.legend(bbox_to_anchor=(1,1))
    ax[0].set_ylabel("generation \n [TWh]")

    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"h2_generationmix_{volume}{name}.pdf",
                bbox_inches='tight')


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

for csv in snakemake.input:
    emissions = pd.concat([emissions, pd.read_csv(csv,index_col=[0,1,2,3])])
    nodal_capacities = pd.concat([nodal_capacities,
                                  pd.read_csv(csv.replace("emissions", "nodal_capacities"),
                                   index_col=[0,1,2], header=[0,1,2,3])], axis=1)
    weighted_prices = pd.concat([weighted_prices,
                                 pd.read_csv(csv.replace("emissions", "weighted_prices"),
                                  index_col=[0], header=[0,1,2,3,4])], axis=1)
    curtailment = pd.concat([curtailment, pd.read_csv(csv.replace("emissions", "curtailment"),
                              index_col=[0,1], header=[0,1,2,3])], axis=1)

    costs = pd.concat([costs, pd.read_csv(csv.replace("emissions", "costs"),
                        index_col=[0,1,2], header=[0,1,2,3])])

    nodal_costs = pd.concat([nodal_costs, pd.read_csv(csv.replace("emissions", "nodal_costs"),
                              index_col=[0,1,2,3], header=[0,1,2,3])], axis=1)
    try:
        h2_cost = pd.concat([h2_cost, pd.read_csv(csv.replace("emissions", "h2_costs"),
                              index_col=[0,1], header=[0,1,2,3])], axis=1)
    except:
        print(f"h2 cost {csv}")
    try:
        emission_rate = pd.concat([emission_rate,
                                   pd.read_csv(csv.replace("emissions", "emission_rate"),
                                    index_col=[0], header=[0,1,2,3])])
    except:
        print(f"emission rate {csv}")
    try:
        h2_gen_mix = pd.concat([h2_gen_mix,
                                pd.read_csv(csv.replace("emissions", "h2_gen_mix"),
                                 index_col=[0,1], header=[0,1,2,3])], axis=1)
    except:
        print(f"h2 gen mix {csv}")
    try:
        attr_emissions = pd.concat([attr_emissions, pd.read_csv(csv.replace("emissions", "attr_emissions"),
                                     index_col=[0,1], header=[0,1,2,3])], axis=1)

    except:
        print(f"attr emissions {csv}")


    supply_energy = pd.concat([supply_energy,
                               pd.read_csv(csv.replace("emissions", "supply_energy"),
                                index_col=[0,1,2], header=[0,1,2,3])], axis=1)

    nodal_supply_energy = pd.concat([nodal_supply_energy,
                                     pd.read_csv(csv.replace("emissions", "nodal_supply_energy"),
                                      index_col=[0,1,2, 3], header=[0,1,2,3])], axis=1)


    cf_s = pd.read_csv(csv.replace("emissions", "cf"),
                                    index_col=0, header=[0,1,2,3,4], parse_dates=True)
    if not cf_s.empty: cf = pd.concat([cf, cf_s], axis=1)

path = snakemake.output["cf_plot"].replace("graphs", "csvs").split("/cf_electrolysis")[0]
final.to_csv(path + "/final_together.csv")
cf.mean().xs(f"{name} H2 Electrolysis", level=4).to_csv(path + "/cf_together.csv")
emissions.to_csv(path + "/emissions_together.csv")
supply_energy.to_csv(path + "/supply_energy_together.csv")
nodal_capacities.to_csv(path + "/nodal_capacities_together.csv")
weighted_prices.to_csv(path + "/weighted_prices_together.csv")
curtailment.to_csv(path + "/curtailment_together.csv")
nodal_costs.to_csv(path + "/nodal_costs_together.csv")
costs.to_csv(path + "/costs_together.csv")
h2_cost.to_csv(path + "/h2_cost_together.csv")
emission_rate.to_csv(path + "/emission_rate_together.csv")
nodal_supply_energy.to_csv(path + "/nodal_supply_together.csv")
h2_gen_mix.to_csv(path + "/h2_gen_mix_together.csv")
attr_emissions.to_csv(path + "/attr_emissions_together.csv")

#%%
a = cf.mean().xs(f"{name} H2 Electrolysis", level=4)

h2_gen_mix = h2_gen_mix.rename(index=lambda x: x.replace("1","")
                               .replace("0","")).droplevel(0)

rename_scenarios = {"res1p0": "annually", "exl1p0": "hourly", "offgrid": "hourly",
                    "grd": "grid",
                    "res1p2": "annually excess 20%", "exl1p2": "hourly excess 20%",
                    "res1p3": "annually excess 30%", "exl1p3": "hourly excess 30%",}

plot_scenarios = {"":["grd", "res1p0", "offgrid","exl1p2"],
                  "wmonthly":["grd", "res1p0", "monthly", "offgrid","exl1p2"],
                  "_sensi_excess":  ["offgrid", "exl1p2", "exl1p3"],
                  "_sensi_excess_annual": ["res1p0", "res1p2", "res1p3"],
                  "_sensi_monthly": ["res1p0", "monthly", "offgrid"],
                  "_sensi_monthly_nohourly": ["res1p0", "monthly"],
                   "_without_hourly": ["grd", "res1p0", "monthly"]
                  }
wished_order = snakemake.config["scenario"]["storage"]
year = snakemake.wildcards.year
country = snakemake.wildcards.zone

#%%
for volume in a.index.get_level_values(2).unique():
    for name, wished_policies in plot_scenarios.items():
        print(name)
        # capacity factors
        plot_cf(a, wished_policies, wished_order, volume, name=name)

        # consequential emissions
        plot_consequential_emissions(emissions, supply_energy,
                                      wished_policies, wished_order,
                                      volume, name=name)

        # attributional emissions
        plot_attributional_emissions(attr_emissions, wished_policies, wished_order,
                                      str(volume), name=name)

        # cost breakdown
        plot_cost_breakdown(h2_cost, wished_policies, wished_order, volume, name=name)

        # shadow prices
        plot_shadow_prices(weighted_prices, wished_policies, wished_order, volume,
                                name=name, carrier="AC")

        # generation mix of H2
        plot_h2genmix(h2_gen_mix, wished_policies, wished_order, str(volume),
                                name=name)

 #%%
name = snakemake.config['ci']['name']
wished_policies = plot_scenarios["wmonthly"]
res = nodal_capacities.loc[nodal_capacities.index.get_level_values(2).isin(["solar", "onwind"])].drop("cfe", axis=1, errors="ignore")
store = nodal_capacities.loc[nodal_capacities.index.get_level_values(2).isin(["H2 Store", "battery"])].drop("cfe", errors="ignore", axis=1)
elec = nodal_capacities.loc[nodal_capacities.index.get_level_values(2).isin(["H2 Electrolysis"])].drop("cfe", errors="ignore", axis=1)
supply_energy.drop("cfe", axis=1, level=0,inplace=True, errors="ignore")
policy_order = [rename_scenarios[x] if x in rename_scenarios.keys() else x
                for x in plot_scenarios["wmonthly"]]
for volume in res.columns.levels[2]:

    caps = res.xs((res_share, volume), level=[1,2], axis=1).xs(f"{name}", level=1).droplevel(0).reindex(wished_policies, level=0, axis=1).fillna(0)
    caps = caps.stack().reindex(wished_policies, axis=1).fillna(0).unstack()
    caps = caps.reindex(wished_order, level=1, axis=1)
    caps.rename(columns=rename_scenarios,
                        level=0,inplace=True)
    caps.rename(index=rename_techs, inplace=True)
    fig, ax = plt.subplots(nrows=1, ncols=len(caps.columns.levels[0]), sharey=True,
                            figsize=(10,1.5))
    for i, policy in enumerate(policy_order):

        (caps[policy].T/1e3).plot(grid=True,
                        title = policy,
                        color=[snakemake.config["tech_colors"][i] for i in caps.index],
                        kind="bar", stacked=True,
                        ax=ax[i],
                        legend=False,
                        width=0.65,
                        )
        ax[i].grid(alpha=0.3)
        ax[i].set_axisbelow(True)
    ax[0].set_ylabel("capacity \n [GW]")
    plt.legend(bbox_to_anchor=(1,1))
    plt.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"capacities_RES_{volume}volume.pdf",
                    bbox_inches='tight')

    caps = elec.xs((res_share, volume), level=[1,2], axis=1).xs(f"{name}", level=1).droplevel(0).reindex(wished_policies, level=0, axis=1).fillna(0)
    caps = caps.stack().reindex(wished_policies, axis=1).fillna(0).unstack()
    caps = caps.reindex(wished_order, level=1, axis=1)
    caps.rename(columns=rename_scenarios,
                        level=0,inplace=True)
    caps.rename(index=rename_techs, inplace=True)
    fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies), sharey=True, figsize=(10,1.5))
    for i, policy in enumerate(policy_order):

        (caps[policy].T/1e3).plot(grid=True,
                        title = policy,
                        color=[snakemake.config["tech_colors"][i] for i in caps.index],
                        kind="bar", stacked=True,
                        ax=ax[i],
                        legend=False,
                        width=0.65,
                        )
        ax[i].grid(alpha=0.3)
        ax[i].set_axisbelow(True)
    ax[0].set_ylabel("capacity \n [GW]")
    plt.legend(bbox_to_anchor=(1,1))
    plt.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"capacities_electrolysis_{volume}volume.pdf",
                    bbox_inches='tight')


    caps = store.xs((res_share, volume), level=[1,2], axis=1).xs(f"{name}", level=1).droplevel(0).reindex(wished_policies, level=0, axis=1).fillna(0)
    caps = caps.stack().reindex(wished_policies, axis=1).fillna(0).unstack()
    caps = caps.reindex(wished_order, level=1, axis=1)
    caps.rename(columns=rename_scenarios,
                        level=0,inplace=True)
    caps.rename(index=rename_techs, inplace=True)
    caps.drop("flexibledemand", level=1, axis=1, inplace=True)
    fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies), sharey=True, figsize=(10,1.5))
    for i, policy in enumerate(policy_order):

        (caps[policy].T/1e3).plot(grid=True,
                        title = policy,
                        color=[snakemake.config["tech_colors"][i] for i in caps.index],
                        kind="bar", stacked=True,
                        ax=ax[i],
                        width=0.65,
                        legend=False
                        )
        ax[i].grid(alpha=0.3)
        ax[i].set_axisbelow(True)
    ax[0].set_ylabel("energy capacity \n [GWh]")
    plt.legend(bbox_to_anchor=(1,1))
    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"capacities_price_{volume}volume.pdf",
                bbox_inches='tight')





    c = curtailment.xs((res_share, volume), level=[1,2], axis=1).reindex(wished_policies, level=0, axis=1).fillna(1)
    res_car = ["solar", "onwind", "all"]
    snakemake.config['tech_colors']["all"] = "b"
    for carrier in res_car:
        d = c.loc[c.index.get_level_values(0).str.contains(carrier)].groupby(level=1).mean().stack().unstack(-2).dropna(axis=1, how="all")
        d[d<0] = 0
        d = d.reindex(wished_policies,axis=1, level=0)
        d = d.reindex(index=wished_order)
        d.rename(columns=rename_scenarios,
                          inplace=True)
        if carrier=="all":
            d = d.droplevel(1, axis=1)
        else:
            d = d.xs(f"{name}", axis=1,level=1)
        d = d.reindex(columns=policy_order)
        fig, ax = plt.subplots(nrows=1, ncols=len(d.columns), sharey=True, figsize=(10,1.5))
        for i, policy in enumerate(d.columns):
            ((1-d[policy])*100).plot(kind="bar", grid=True, ax=ax[i], color=snakemake.config['tech_colors'][carrier],
                                      width=0.65,
                                                  title=policy, legend=False)
            ax[i].grid(alpha=0.3)
            ax[i].set_axisbelow(True)
        ax[0].set_ylabel("curtailment \n [%]")
        # plt.xlabel("price \n [Eur/kg$_{H2}$]")
        # plt.legend(bbox_to_anchor=(1,1))
        fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"curtailment_{volume}volume_{carrier}.pdf",
                    bbox_inches='tight')

    traffic = supply_energy.loc["AC"].droplevel(0).loc[["export0", "import1"]][wished_policies].xs((res_share, volume), level=[1,2], axis=1).fillna(0)
    traffic.rename(columns=rename_scenarios,
                      inplace=True)
    traffic = traffic.reindex(wished_order, level=1, axis=1)
    traffic.rename(index=lambda x:rename_techs[x.replace("0","").replace("1","")], inplace=True)
    fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies), sharey=True, figsize=(10,1.5))
    for i, policy in enumerate(traffic.columns.levels[0]):
        (traffic[policy]/1e6).T.plot(kind="bar", grid=True, ax=ax[i], color=[snakemake.config['tech_colors'][i] for i in traffic.index],
                                  width=0.65, stacked=True,
                                              title=policy, legend=False)
        ax[i].grid(alpha=0.3)
        ax[i].set_axisbelow(True)
    ax[0].set_ylabel("energy \n [TWh]")
    plt.legend()
    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"export_import_{volume}volume.pdf",
                bbox_inches='tight')

    plot_duration_curve(cf, f"{name} H2 Electrolysis", wished_policies, wished_order, volume,
                        name="electrolysis")
    plot_duration_curve(cf, f"{name} H2 Store", wished_policies, wished_order, volume,
                        name="store")

#%%
# plot_nodal_balances(nodal_supply_energy)
# #%%
# base_year = snakemake.wildcards.year
# base_ct = snakemake.wildcards.zone
# tot_generation = {}
# scenarios =  [("DE", "2025"),  ("DE", "2030"), ("NL", "2025")] #  [("DE", "2025"), ("DE", "2030"), ("NL", "2025")]
# for scenario in scenarios:
#     ct = scenario[0]
#     year = scenario[1]
#     res_share = str(snakemake.config[f"res_target_{year}"][ct])
#     input_path = snakemake.output.csvs_nodal_supply_energy.replace(base_year, year).replace(f"/{base_ct}/", f"/{ct}/")
#     tot_generation[scenario] = pd.read_csv(input_path, index_col=[0,1,2, 3], header=[0,1,2,3]).xs(ct, level=2).xs(res_share, level=1, axis=1)
# #%%
# tot_generation = pd.concat(tot_generation, axis=1)
# co2_carriers = ["co2", "co2 stored", "process emissions"]

# balances = {i.replace(" ","_"): [i] for i in tot_generation.index.levels[0]}
# balances["energy"] = [i for i in tot_generation.index.levels[0] if i not in co2_carriers]


# k = "AC"
# v = ["AC"]

# df = tot_generation.loc[v]
# df = df.groupby(df.index.get_level_values(2)).sum()

# #convert MWh to TWh
# df = df / 1e6

# #remove trailing link ports
# df.index = [i[:-1] if ((i not in ["co2", "NH3", "H2"]) and (i[-1:] in ["0","1","2","3"])) else i for i in df.index]

# # df = df.groupby(df.index.map(rename_techs)).sum()
# df = df.groupby(df.index).sum()


# df = df.xs(("ref", volume, "flexibledemand"), level=[2,3,4], axis=1)

# to_drop = df.index[df.abs().max(axis=1) < 1] # snakemake.config['plotting']['energy_threshold']/10]

# print("dropping")

# print(df.loc[to_drop])

# df = df.drop(to_drop)


# df.drop(["AC", "DC"], errors="ignore",inplace=True)
# df.rename(index=rename_techs,inplace=True)
# df = df.groupby(df.index).sum()
# # df.columns = [f"{snakemake.wildcards.zone}-{snakemake.wildcards.year}"]

# a = df[df>0].dropna(axis=0, how="all")


# wished_tech_order = pd.Index(["electricity", "hydro", "offshore wind",
#                       "onshore wind", "solar PV", "biomass CHP",
#                       'CCGT', 'coal', "lignite", "oil"])
# order = wished_tech_order.intersection(a.index).union(a.index.difference(wished_tech_order))
# a = a.reindex(order)
# a = (a/a.sum())*100
# a.loc["offshore wind",:] = a[a.index.str.contains("offshore")].sum()
# a.drop(["offshore wind AC", "offshore wind DC"], errors="ignore", inplace=True)
# fig, ax = plt.subplots(figsize=(10,1.5))
# a.T.plot(kind="bar", ax=ax, stacked=True, grid=True,
#                       color=[snakemake.config['tech_colors'][i] for i in a.index],
#                       width=0.65)
# ax.grid(alpha=0.3)
# ax.set_axisbelow(True)
# plt.ylabel("share of generation \n [%]")
# plt.legend(bbox_to_anchor=(1,1))
# fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"energy-{k}-noimports.pdf",
#             bbox_inches='tight')


# n = pypsa.Network("/home/lisa/mnt/247-cfe/results/remove_one_snapshot/networks/10/2025/DE/p1/res1p0_p0_3200volume_htank.nc",
#                   override_component_attrs=override_component_attrs())
# m = pypsa.Network("/home/lisa/mnt/247-cfe/results/remove_one_snapshot/networks/10/2025/DE/p1/res1p0_p0_3200volume_mtank.nc",
#                   override_component_attrs=override_component_attrs())
# htank = n.generators_t.p.loc[:,n.generators.bus==f"{name}"].groupby(n.generators.carrier, axis=1).sum()[["onwind", "solar"]]
# htank = pd.concat([htank], keys=["htank"], axis=1)
# mtank = m.generators_t.p.loc[:,n.generators.bus==f"{name}"].groupby(n.generators.carrier, axis=1).sum()[["onwind", "solar"]]
# mtank = pd.concat([mtank], keys=["mtank"], axis=1)
# together = pd.concat([mtank, htank], axis=1)
# fig, ax = plt.subplots()
# together.groupby(together.index.month).sum().plot(ax=ax)
# together.groupby(together.index.month).sum().groupby(level=0, axis=1).sum().plot(ax=ax)
# plt.legend(bbox_to_anchor=(1,1))
