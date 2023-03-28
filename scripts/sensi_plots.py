#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:15:45 2023

@author: lisa
"""
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
from plot_offtake import rename_scenarios, rename_techs

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        import os
        os.chdir("/home/lisa/mnt/hourly_vs_annually/scripts")
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('plot_offtake', configfiles="/home/lisa/mnt/hourly_vs_annually/results/two_step_resshare/configs/config.yaml",
                                   palette='p1',
                                   zone='DE', year='2025',  participation='10',
                                   policy="ref")
        os.chdir("/home/lisa/mnt/hourly_vs_annually/")

LHV_H2 = 33.33 # lower heating value [kWh/kg_H2]
volume = 3200
path = "/home/lisa/Documents/own_projects/green_h2/figures/new/graphs/"

snakemake.config['tech_colors']["solar local"] = "#ffdd78"
snakemake.config['tech_colors']["onwind local"] = "#bae3f9"
for k,v in rename_techs.items():
    snakemake.config['tech_colors'][v] = snakemake.config['tech_colors'][k]

name = snakemake.config['ci']['name']

year = snakemake.wildcards.year
country = snakemake.wildcards.zone
res_share = str(snakemake.config[f"res_target_{year}"][country])
#%%
# consequential emissions NL and DE ------------------
supply_energy_NL = pd.read_csv("/home/lisa/mnt/hourly_vs_annually/results/two_steps_withhout_h2demand/csvs/2025/NL/p1/supply_energy_together.csv",
                            index_col=[0,1,2], header=[0,1,2,3])

supply_energy_NL = pd.concat([supply_energy_NL], keys=["NL"], axis=1)
supply_energy_DE = pd.read_csv("/home/lisa/mnt/hourly_vs_annually/results/two_steps_withhout_h2demand/csvs/2030/DE/p1/supply_energy_together.csv",
                            index_col=[0,1,2], header=[0,1,2,3])

supply_energy_DE = pd.concat([supply_energy_DE], keys=["DE"], axis=1)

supply_energy = pd.concat([supply_energy_NL, supply_energy_DE], axis=1)

supply_energy = supply_energy.xs(str(volume), level=3, axis=1)

wished_policies = ['grd', 'res1p0', 'monthly']
for policy in wished_policies:
    wished_columns = supply_energy.columns.get_level_values(1)==policy
    supply_energy.loc[:, wished_columns] = (supply_energy.xs(policy, axis=1, level=1) - supply_energy.xs("ref", axis=1, level=1)).values



def plot_consequential_emissions_c(supply_energy, wished_policies,
                                 wished_order, volume, name=""):

    emissions_s = supply_energy.droplevel([2], axis=1).loc["co2"]
    emissions_s = emissions_s.rename(index=lambda x:x.replace("2",""))
    emissions_s.rename(rename_scenarios,level=1,inplace=True, axis=1)
    emissions_s = emissions_s.loc["links"].drop("nuclear")


    nice_names = [rename_scenarios[scen] if scen in rename_scenarios.keys()
                  else scen for scen in wished_policies]

    y_min = 0
    y_max = 0

    # annually produced H2 in [t_H2/a]
    produced_H2 = volume*8760 / LHV_H2
    fig, ax = plt.subplots(nrows=1, ncols=2*len(wished_policies), sharey=True,figsize=(10,1.5))

    for i, policy in enumerate(nice_names*2):
        ct = "NL" if i in [0,1] else "DE"
        print(ct, i, policy)
        em_p = emissions_s[ct][policy]/ produced_H2

        em_p.sum().rename("net total").plot(ax=ax[i], lw=0, marker="_", color="black",
                                            markersize=15, markeredgewidth=2)

        em_p.T.plot(kind="bar", stacked=True, grid=True, ax=ax[i], title=policy,
                            width=0.65, legend=False,
                            color=[snakemake.config['tech_colors'][i] for i in em_p.index])


        ax[i].set_xlabel("")
        ax[i].grid(alpha=0.3)
        ax[i].set_axisbelow(True)
        if em_p[em_p<0].sum().min()<y_min: y_min = em_p[em_p<0].sum().min()
        if em_p[em_p>0].sum().max()>y_max: y_max = em_p[em_p>0].sum().max()


    ax[0].set_ylim([1.1*y_min, 1.1*y_max])
    ax[0].set_ylabel("consequential emissions \n [kg$_{CO_2}$/kg$_{H_2}$]")
    ax[0].text(1.8, 1.3, 'Netherlands 2025', horizontalalignment='center',
               verticalalignment='center', transform=ax[0].transAxes, fontsize=14)
    ax[0].text(5.4, 1.3, 'Germany 2030', horizontalalignment='center',
               verticalalignment='center', transform=ax[0].transAxes, fontsize=14)
    plt.legend(fontsize=9, bbox_to_anchor=(1,1))

    fig.savefig(path+ f"consequential_emissions_by_carrier_{volume}_cleanness.pdf",
                bbox_inches="tight")



def plot_cf_shares(df, wished_policies, wished_order, volume, name=""):
    # df = df.xs(res_share, level=1)
    df = df[~df.index.duplicated()]
    cf_elec = df.loc[wished_policies].xs(volume, level=2)
    cf_elec.sort_index(level=1, inplace=True)
    cf_elec = cf_elec.reindex(wished_order, level=2)
    cf_elec.rename(index=rename_scenarios,
                   level=0,inplace=True)
    for policy in cf_elec.index.levels[0]:
        shares = cf_elec.index.get_level_values(1).unique()
        fig, ax = plt.subplots(nrows=1, ncols=len(shares), sharey=True,
                               figsize=(10,1.5))

        for i, share in enumerate(shares):
            if share==res_share:
                title=f"base\n {share}"
            else:
                title=share
            cf_elec.loc[policy, share].plot(kind="bar", grid=True, ax=ax[i], title=title,
                                     width=0.65)

            ax[i].set_xlabel("")
            ax[i].grid(alpha=0.3)
            ax[i].set_axisbelow(True)
        ax[0].set_ylabel("capacity factor")
        fig.savefig(snakemake.output.cf_plot.split(".pdf")[0] + f"{policy}_resshare.pdf",
                    bbox_inches="tight")

    fig.savefig(snakemake.output.cf_plot,
                bbox_inches="tight")


def plot_consequential_emissions_share(emissions, supply_energy, wished_policies,
                                 wished_order, volume, name=""):
    # consequential emissions
    emissions = emissions[~emissions.index.duplicated()]
    emissions_v = emissions.loc[wished_policies+["ref"]].xs(float(volume), level=2)
    emissions_v.sort_index(level=1,inplace=True)
    emissions_v = emissions_v.reindex(wished_order, level=2)
    emissions_v.rename(index=rename_scenarios,
                       level=0,inplace=True)
    emissions_s = (supply_energy.loc["co2"].droplevel(0)[wished_policies+["ref"]]
                   .xs(volume,level=2, axis=1))
    emissions_s.sort_index(level=1,inplace=True)
    emissions_s = emissions_s.reindex(wished_order, level=2, axis=1)
    emissions_s = emissions_s[emissions_s>0].dropna(axis=0).rename(index=lambda x:x.replace("2",""))
    emissions_s.rename(rename_scenarios,level=0,inplace=True, axis=1)

    nice_names = [rename_scenarios[scen] if scen in rename_scenarios.keys()
                  else scen for scen in wished_policies]
    for policy in nice_names:
        shares = emissions_v.index.get_level_values(1).unique()
        fig, ax = plt.subplots(nrows=1, ncols=len(shares), sharey=True,figsize=(10,1.5))
        for i, share in enumerate(shares):
            # annually produced H2 in [t_H2/a]
            produced_H2 = float(volume)*8760 / LHV_H2
            em_p = emissions_v.loc[policy, share].sub(emissions_v.loc["ref", share].mean())/ produced_H2

            if str(share)==res_share:
                title=f"base\n {share}"
            else:
                title=share

            em_p.iloc[:,0].plot(kind="bar", grid=True, ax=ax[i], title=title,
                                width=0.65)
            ax[i].set_xlabel("")
            ax[i].grid(alpha=0.3)
            ax[i].set_axisbelow(True)
        ax[0].set_ylabel("consequential emissions \n [kg$_{CO_2}$/kg$_{H_2}$]")
        fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0]+ f"consequential_emissions_{policy}_RESshare.pdf",
                    bbox_inches="tight")


    for i, policy in enumerate(nice_names):
        shares = emissions_s.columns.get_level_values(1).unique()
        y_min = 0
        y_max = 0
        fig, ax = plt.subplots(nrows=1, ncols=len(shares), sharey=True,figsize=(10,1.5))
        for i, share in enumerate(shares):
            # annually produced H2 in [t_H2/a]
            produced_H2 = float(volume)*8760 / LHV_H2
            em_p = emissions_s[policy, share].sub(emissions_s["ref", share])/ produced_H2

            if str(share)==res_share:
                title=f"base\n {share}"
            else:
                title=share


            em_p.sum().rename("net total").plot(ax=ax[i], lw=0, marker="_", color="black",
                                                markersize=8, markeredgewidth=2)
            # em_p.sum().rename("net total").plot(ax=ax[i], kind="bar", color="black",
            #                                     width=0.25, position=-1)
            em_p.T.plot(kind="bar", stacked=True, grid=True, ax=ax[i], title=title,
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
        ax[i].legend(loc="upper left", bbox_to_anchor=(1,1))
        fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0]+ f"consequential_emissions_by_carrier_{policy}-RESshare.pdf",
                    bbox_inches="tight")


def plot_attributional_emissions_share(attr_emissions, wished_policies, wished_order, volume, name=""):
    for consider_import in ['no imports', 'with imports']:
        attr_emissions = attr_emissions[~attr_emissions.index.duplicated()]
        em_r = attr_emissions.loc[consider_import].xs(volume, level=2, axis=1)
        em_r = em_r.stack().stack().reindex(wished_policies,axis=1).fillna(0).unstack().unstack()
        em_r = em_r.reindex(wished_order, axis=1, level=2)
        em_r.rename(columns=rename_scenarios,
                           level=0,inplace=True)
        nice_names = [rename_scenarios[scen] if scen in rename_scenarios.keys()
                      else scen for scen in wished_policies]
        for policy in nice_names:
            shares = em_r.columns.levels[1]
            # figsize=(4.5,3.5) if len(wished_policies)==2  else  (9,3.5)
            fig, ax = plt.subplots(nrows=1, ncols=len(shares), sharey=True,figsize= (len(shares)*(4.5/2),3.5),
                                   )

            for i, share in enumerate(shares):
                if str(share)==res_share:
                    title=f"base\n {share}"
                else:
                    title=share

                em_r[policy, share].T.plot(kind="bar", stacked=True, grid=True, ax=ax[i],
                                    title=title,
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
            ax[len(shares)-1].text(x= 4.6, y=0.8, s='carbon intensity of\nblue hydrogen')
            ax[len(shares)-1].text(x= 4.6, y=9.8, s='carbon intensity of\ngrey hydrogen')
            ax[len(shares)-1].text(x=4.6, y=2.8, s='EU threshold for \nlow-carbon hydrogen')
            if consider_import == "with imports":
                suffix = ""
            else:
                suffix = "_noimports"
            ax[len(share)-1].legend(ncol=5, bbox_to_anchor=(5,1.2), loc="upper left")
            # plt.legend(ncol=2) # bbox_to_anchor=(1,0.85), loc="upper left")
            plt.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"attributional_emissions_{policy}-RESshare_{suffix}.pdf",
                        bbox_inches='tight')

def plot_cost_breakdown_shares(h2_cost, wished_policies, wished_order, volume, name=""):
    h2_cost = h2_cost[~h2_cost.index.duplicated()]
    costb = h2_cost.xs(volume, level=2, axis=1).droplevel(0)
    costb = costb.stack().stack().reindex(wished_policies, axis=1).fillna(0).unstack().unstack()
    costb.sort_index(level=1,axis=1,inplace=True)
    costb = costb.reindex(wished_order, level=2, axis=1)
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

    for policy in singlecost.columns.get_level_values(0).unique():
        shares = singlecost.columns.get_level_values(1).unique()
        fig, ax = plt.subplots(nrows=1, ncols=len(shares), sharey=True,figsize=(len(shares)*2.25,3.5))
        for i, share in enumerate(shares):
            cost_p = singlecost[policy,share]
            if share==res_share:
                title=f"base\n {share}"
            else:
                title=share
            cost_p.sum().rename("net total").plot(ax=ax[i], marker="_",
                                                                  lw=0, color="black",
                                                                  markersize=10, markeredgewidth=2)
            cost_p.T.plot(kind="bar", stacked=True, ax=ax[i], title=title,
                     color=[snakemake.config['tech_colors'][i] for i in singlecost.index],
                     grid=True, legend=False,  width=0.65)
            ax[i].grid(alpha=0.3)
            ax[i].set_axisbelow(True)
        not_grd = singlecost.columns!=(    'grid',        'nostore')
        if policy!="grid":
            y_max = singlecost[policy][singlecost[policy]>0].sum().max()*1.1
        else:
            y_max = singlecost[policy][singlecost[policy]>0].sum().drop("nostore", level=1).max()*1.1
        ax[0].set_ylim([singlecost[policy][singlecost[policy]<0].sum().min()*1.1,
                        y_max])
        ax[0].set_ylabel("cost \n [Euro/kg$_{H_2}$]")
        plt.legend(bbox_to_anchor=(1,1))
        fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"costbreakdown_{policy}.pdf",
                    bbox_inches='tight')

#%%% res share
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
wished_policies = plot_scenarios["wmonthly"]
plot_cf_shares(a, wished_policies, wished_order, volume)
plot_consequential_emissions_share(emissions, supply_energy, wished_policies,
                                  wished_order, volume, name="")
plot_attributional_emissions_share(attr_emissions, wished_policies, wished_order,
                                    volume, name="")
plot_cost_breakdown_shares(h2_cost, wished_policies, wished_order, volume,
                            name="")
