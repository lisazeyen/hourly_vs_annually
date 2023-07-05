#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:06:34 2023

@author: lisa
"""

import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from _helpers import override_component_attrs

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


path = "/home/lisa/Documents/hourly_vs_annually/results/additional_graphics/"
input_network = "/home/lisa/mnt/hourly_vs_annually_copy/results/DE_NL_avoid_suboptimal/networks/2025/DE/p1/res1p0_p0_3200volume_nostore.nc"
ref_network = "/home/lisa/mnt/hourly_vs_annually_copy/results/DE_NL_avoid_suboptimal/base/2025/DE/p1/base_p0_3200volume.nc"

n = pypsa.Network(input_network,
                  override_component_attrs=override_component_attrs())

# marginal costs by carrier
marginal_cost_links = n.links.marginal_cost.groupby(n.links.carrier).mean()
marginal_cost_gens = n.generators.marginal_cost.groupby(n.generators.carrier).mean()
marginal_cost_gens.rename({"uranium": "nuclear", "gas" : "OCGT"}, inplace=True)
marginal_cost_gens["CCGT"]  = marginal_cost_gens["OCGT"]
# fuel cost
fuel_cost = marginal_cost_gens.div(n.links.efficiency.groupby(n.links.carrier).mean())


# merit order
m_order = (marginal_cost_links + fuel_cost.fillna(0))
m_order = m_order.fillna(marginal_cost_gens).sort_values()
wished = ['ror', 'solar', 'onwind', 'offwind',
       'offwind-ac', 'offwind-dc', 'urban central solid biomass CHP',
       'nuclear', 'CCGT', 'coal', 'lignite', 'OCGT', 'oil']
m_order = m_order.loc[wished]
offwind = m_order.groupby(m_order.index.str.contains("offwind")).mean().loc[True]
m_order.drop(m_order.index[m_order.index.str.contains("offwind")], inplace=True)
m_order.loc["offwind"] = offwind
m_order.sort_values(inplace=True)
m_order.rename(index={"urban central solid biomass CHP": "biomass CHP"}, inplace=True)

fig, ax = plt.subplots(figsize=(10,1.5))
bars = m_order.plot(kind="bar",color=[snakemake.config['tech_colors'][i]
                               for i in  m_order.index], ax=ax,
             title="marginal cost by carrier")
ax.set_ylabel("Eur/MWh")
ax.set_xlabel("")
ax.set_ylim([0, 1.3*m_order.max()])

ax.grid(axis="y", alpha=0.5)
ax.set_axisbelow(True)

for bar in bars.containers[0]:
    height = bar.get_height()
    ax.annotate('{:.1f}'.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.savefig(path + "merit_order.pdf", bbox_inches="tight")
# %%
demand_elec = n.links_t.p0["CI H2 Electrolysis"]
local_gen = n.generators_t.p.loc[:, n.generators.bus=="CI"]
residual_load = (demand_elec - local_gen.sum(axis=1)).mul(n.snapshot_weightings.generators)

((residual_load/1e3).sort_values(ascending=False)
 .reset_index(drop=True)
 .rename(index=lambda x: 3*x)
 .plot(grid=True, title="electrolysis demand  - local VRES"))
plt.axhline(y=0, color="grey")
plt.xlabel("hours")
plt.ylabel("residual load \n [GWh]")
plt.savefig(path + "residual_load_DE_annual_2025_withUC_nostore.pdf", bbox_inches="tight")
#%% christophs plot
vre = ['CI onwind', 'CI solar']
def get_time_series(n, vre):
    df = pd.DataFrame(index=n.snapshots)

    df["vre_available"] = n.generators_t.p_max_pu[vre].multiply(n.generators.p_nom_opt.loc[vre],axis=1).sum(axis=1)
    df["vre_dispatched"] = n.generators_t.p[vre].sum(axis=1)
    df["vre_curtailed"] = abs(df["vre_available"] - df["vre_dispatched"])
    df["load"] = n.links_t.p0["CI H2 Electrolysis"]
    df["electrolysis"] = n.links_t.p0["CI H2 Electrolysis"]
    df["residual_load"] = df["load"] - df["vre_dispatched"]  # Chris had this different, why?
    df["battery_charge"] = n.links_t.p0["CI battery charger"]
    df["battery_discharge"] = n.links_t.p1["CI battery discharger"]
# add other time series from network results
    return df

def get_residual(df):
    residual = pd.DataFrame(index=range(len(n.snapshots)))

    residual = df.sort_values('residual_load',ascending=False)
    residual["vre_curtailed"] *= -1
    residual["vre_dispatched"] *= -1

    return residual

df=get_time_series(n,vre)
residual=get_residual(df)
wished = ["battery_discharge","electrolysis","battery_charge","vre_curtailed", "vre_dispatched"]
residual = residual.reset_index(drop=True).rename(index=lambda x: 3*x)

fig, ax = plt.subplots()
(residual["residual_load"]/1e3).plot(ax=ax,color="k",linewidth=2)
(residual[wished]/1e3).plot(kind="area", 
                            linewidth=0,stacked=True,
                            ax=ax,
                            # use_index=False,
                            title="residual load = electrolysis  demand - VRE feed-in")
plt.xlabel("hours")
plt.ylabel("residual load \n [GWh]")
plt.legend(ncol=2)
plt.savefig(path + "residual_load_christoph_plot_DE_2025_annual_withUC_nostore.pdf", bbox_inches="tight")
#%%
from summarise_offtake import assign_locations

def get_supply(network, carrier="AC", name="test"):
    n = network.copy()

    buses = n.buses.index[n.buses.carrier.str.contains(carrier)]

    supply = pd.DataFrame(index=n.snapshots)
    for c in n.iterate_components(n.branch_components):
        n_port = 4 if c.name == "Link" else 2
        for i in range(n_port):
            supply = pd.concat(
                (
                    supply,
                    (-1)
                    * c.pnl["p" + str(i)]
                    .loc[:, c.df.index[c.df["bus" + str(i)].isin(buses)]]
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )

    for c in n.iterate_components(n.one_port_components):
        comps = c.df.index[c.df.bus.isin(buses)]
        supply = pd.concat(
            (
                supply,
                ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
                .groupby(c.df.carrier, axis=1)
                .sum(),
            ),
            axis=1,
        )

    # supply = supply.groupby(rename_techs_tyndp, axis=1).sum()
    supply = supply.groupby(level=0, axis=1).sum()

    both = supply.columns[(supply < 0.0).any() & (supply > 0.0).any()]

    positive_supply = supply[both]
    negative_supply = supply[both]

    positive_supply[positive_supply < 0.0] = 0.0
    negative_supply[negative_supply > 0.0] = 0.0

    supply[both] = positive_supply

    suffix = " charging"

    negative_supply.columns = negative_supply.columns + suffix

    supply = pd.concat((supply, negative_supply), axis=1)

    threshold = 1e3

    to_drop = supply.columns[(abs(supply) < threshold).all()]

    if len(to_drop) != 0:
        print(f"Dropping {to_drop.tolist()} from supply")
        supply.drop(columns=to_drop, inplace=True)

    supply.index.name = None

    supply = supply / 1e3

    supply.rename(
        columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
    )
    supply.columns = supply.columns.str.replace("residential ", "")
    supply.columns = supply.columns.str.replace("services ", "")
    supply.columns = supply.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index(
        [
            "electric demand",
            "transmission lines",
            "hydroelectricity",
            "hydro reservoir",
            "run of river",
            "pumped hydro storage",
            "CHP",
            "onshore wind",
            "offshore wind",
            "solar PV",
            "solar thermal",
            "building retrofitting",
            "ground heat pump",
            "air heat pump",
            "resistive heater",
            "OCGT",
            "gas boiler",
            "gas",
            "natural gas",
            "methanation",
            "hydrogen storage",
            "battery storage",
            "hot water storage",
        ]
    )

    new_columns = preferred_order.intersection(supply.columns).append(
        supply.columns.difference(preferred_order)
    )

    supply = supply.groupby(supply.columns, axis=1).sum()

    return supply

def plot_series(supply, df):
    both = supply.columns[(supply < 0.0).any() & (supply > 0.0).any()]

    positive_supply = supply[both]
    negative_supply = supply[both]

    positive_supply[positive_supply < 0.0] = 0.0
    negative_supply[negative_supply > 0.0] = 0.0

    supply[both] = positive_supply

    suffix = " reduction"

    for i in negative_supply.columns:

        snakemake.config['tech_colors'][i + suffix] = snakemake.config['tech_colors'][i]

    snakemake.config['tech_colors']["PHS charging"] = snakemake.config['tech_colors']["PHS charger"]

    negative_supply.columns = negative_supply.columns + suffix

    supply = pd.concat((supply, negative_supply), axis=1)

    supply = pd.concat([supply, df['residual_load']/1e3], axis=1).sort_values("residual_load", ascending=False)

    supply = supply.reset_index(drop=True).rename(lambda x: 3*x)

    fig, ax = plt.subplots()
    fig.set_size_inches((8, 5))

    (supply["residual_load"]).plot(ax=ax,color="k",linewidth=2)

    a = supply.loc[:,~supply.columns.str.contains("residual_load")]

    (
        a.plot(
            ax=ax,
            kind="area",
            stacked=True,
            linewidth=0.0,
            color=[snakemake.config['tech_colors'][i]
                                           for i in  a.columns],
            title="residual load = electrolysis  demand - VRE feed-in"
        )
    )

    handles, labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    new_handles = []
    new_labels = []

    for i, item in enumerate(labels):
        if (suffix not in item) and ("charging" not in item):
            new_handles.append(handles[i])
            new_labels.append(labels[i])

    ax.legend(new_handles, new_labels, ncol=3, loc="upper left", frameon=False)
    # ax.set_xlim([start, stop])
    ax.set_ylim([residual.residual_load.min()/1e3*1.1, supply.sum(axis=1).max()*1.1])
    ax.grid(True)
    ax.set_ylabel("Power [GW]")
    fig.tight_layout()
    fig.savefig(path + "diff_reference_christoph_plot_DE_annual_2025_withUC_flexible.pdf", bbox_inches="tight")
#%%
snakemake.config["tech_colors"]["electric demand"] = snakemake.config["tech_colors"]["electricity"]
ref = pypsa.Network(ref_network,
                  override_component_attrs=override_component_attrs())
supply = get_supply(n, carrier="AC")
supply_ref = get_supply(ref, carrier="AC")
diff = supply - supply_ref.reindex(columns=supply.columns).fillna(0)
diff.drop(diff.loc[:,abs(diff.sum())<1].columns, inplace=True, axis=1)
snakemake.config['tech_colors']["PHS charging"] = snakemake.config['tech_colors']["PHS"]
plot_series(diff, df)
