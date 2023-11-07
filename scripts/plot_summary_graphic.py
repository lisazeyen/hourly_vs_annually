#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:24:10 2023

@author: lisa
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        import os
        os.chdir("/home/lisa/Documents/hourly_vs_annually/scripts")
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('plot_offtake', palette='p1',
                                   zone='DE', year='2025',  participation='10',
                                   policy="ref")
        os.chdir("/home/lisa/Documents/")

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
    
    
# Define marker types for policies
policy_markers = {
    'grd': 's',      # square marker
    'offgrid': 'o',  # circle marker
    'monthly': '^',   # triangle marker
    'res1p0': '*',
    'res1p2': 'p',
    'res1p3': '4',
    "exl1p2": 'h',
    "exl1p3": '8'
}

# Define line styles for storage types
storage_styles = {
    'flexibledemand': 'solid',   # solid line
    'mtank': 'bottom',
    'nostore': 'dotted'     # dotted line
}

storage_marker_styles = {
    'flexibledemand': 'full',   # filled marker
    'mtank': 'bottom',
    'nostore': 'none'          # unfilled marker
}
rename_scenarios = {"res1p0": "annually", "exl1p0": "hourly", "offgrid": "hourly",
                    "grd": "grid", "monthly": "monthly",
                    "res1p2": "annually excess 20%", "exl1p2": "hourly excess 20%",
                    "res1p3": "annually excess 30%", "exl1p3": "hourly excess 30%",}

def plot_multiindex_df(df, name):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique values for countries, policies, and storage types
    countries = df.index.get_level_values(0).unique()
    policies = df.index.get_level_values(1).unique()
    storages = df.index.get_level_values(2).unique()
    
    # Color map for countries
    # Colorblind-friendly color palette
    cb_palette = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', 
                  '#984ea3', '#999999', '#e41a1c', '#dede00']
    
    # Map each country to a color in the palette
    colors = {country: cb_palette[i % len(cb_palette)] for i, country in enumerate(countries)}

    
    # Plot each line
    for (country, policy, storage), df in to_plot.groupby(level=[0, 1, 2]):
        # Select data for each line
        data = df.loc[(country, policy, storage)]
        ax.plot(data['h2_cost'], data['emissions'], 
                color=colors[country], 
                marker=policy_markers[policy], 
                markersize=8,
                fillstyle=storage_marker_styles[storage],
                linestyle="", # storage_styles[storage],
                label=f"{country}, {policy}, {storage}")
    
    # Create legend handles for policies
    policy_handles = [mlines.Line2D([], [], color='black', marker=policy_markers[policy], linestyle='None',
                             markersize=10, label=rename_scenarios[policy])
                      for policy in wished_policies]
    # Create legend handles for storage types
    storage_handles = [mlines.Line2D([], [], color='gray', marker='v', linestyle='None',
                             markersize=10, fillstyle=storage_marker_styles[storage],
                             label=storage)
                      for storage in wished_stores]
    
    # Create legend handles for countries
    country_handles = [mlines.Line2D([], [], color=colors[country],
                                     marker='s', linestyle='None',
                                     label=country) 
                       for country in countries]
    
    # Combine handles
    handles = country_handles + policy_handles + storage_handles
    
    # Add legend to the plot
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    
    # Set the labels for the axes
    ax.set_xlabel("cost \n [Euro/kg$_{H_2}$]")
    ax.set_ylabel("consequential emissions \n [kg$_{CO_2}$/kg$_{H_2}$]")
    # Set grid
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.7)

    # color the background
    # Highlight the specified area with light green
    ax.axhspan(ymin=ax.get_ylim()[0], ymax=0, xmin=0, xmax=4/ax.get_xlim()[1],
               color='lightgreen', alpha=0.3)
    
    # Get the current limits
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    
    # Calculate position for the text
    x_pos = 3  # Halfway on the x-axis span that's colored green
    y_pos = y_lim[0] + 1.5  # Halfway from the bottom of the y-axis to y=0
    
    # Add the text
    ax.text(x_pos, y_pos, 'low emissions +\n low cost', color='green',
            ha='center', va='center', alpha=0.5)
        
    plt.savefig(f"/home/lisa/Documents/hourly_vs_annually/results/summary_cost_emissions_{year}_{name}.pdf",
                bbox_inches="tight")
        
#%%
compare_p = "offgrid"
emissions = {}
h2_cost = {}
volume = "3200"
country_dict = {"DE-2025": "gas_price_35_nocrossover",
                "DE-2030": "gas_price_35_nocrossover",
                "NL": "gas_price_35_nocrossover",
                "PL": "gas_price_35_highersolver",
                "CZ": "gas_price_35_highersolver",
                "PT": "gas_price_35_highersolver",
                "ES": "gas_price_35_highersolver"}

for ct, run_dir in country_dict.items(): 
    year=2030 if ct == "DE-2030" else 2025
    
    ct_y = ct.split("-")[0]
    path = f"/home/lisa/Documents/hourly_vs_annually/results/{run_dir}/csvs/{year}/{ct_y}/p1/"
    res_share = str(snakemake.config[f"res_target_{year}"][ct_y])
    
    # emissions
    emissions_v = pd.read_csv(path + "emissions_together.csv",
                                index_col=[0,1,2,3]
                                ).xs((float(res_share), int(volume)), level=[1,2])
    produced_H2 = float(volume)*8760 / LHV_H2
    em_p = emissions_v.sub(emissions_v.loc[compare_p].mean())/ produced_H2
    emissions[ct] = em_p
    
    # hydrogen production cost
    costb  = pd.read_csv(path + "h2_cost_together.csv",
                          index_col=[0,1], header=[0,1,2,3]
                          ).xs((res_share, volume), level=[1,2], axis=1)
    singlecost = costb.drop(["H2 cost", "offtake H2"], level=1, errors="coerce").div(float(volume)*8760)
    # convert EUR/MWh_H2 in Eur/kg_H2
    singlecost = singlecost / 1e3 * LHV_H2
    h2_cost[ct] = singlecost
#%%
emissions_all = pd.concat(emissions, axis=1).droplevel(1, axis=1)
h2_cost_all = pd.concat(h2_cost, axis=1)

summed_cost = h2_cost_all.sum()
em = emissions_all.unstack(level=0).unstack()
tot = pd.concat([summed_cost, em], axis=1)
tot.columns = ["h2_cost", "emissions"]
#%%
plot_scenarios = {"":["grd", "res1p0", "offgrid","exl1p2"],
                  "_no_grid":["res1p0", "monthly", "offgrid","exl1p2"],
                  "wmonthly":["grd", "res1p0", "monthly", "offgrid","exl1p2"],
                  "_sensi_excess":  ["offgrid", "exl1p2", "exl1p3"],
                  "_sensi_excess_annual": ["res1p0", "res1p2", "res1p3"],
                  "_sensi_monthly": ["res1p0", "monthly", "offgrid"],
                  "_sensi_monthly_nohourly": ["res1p0", "monthly"],
                   "_without_hourly": ["grd", "res1p0", "monthly"]
                  }


wished_stores = ["flexibledemand",  "mtank", "nostore"]

for name, wished_policies in plot_scenarios.items():
    policy_b = tot.index.get_level_values(1).isin(wished_policies)
    store_b = tot.index.get_level_values(2).isin(wished_stores)
    
    to_plot = tot[policy_b & store_b]
    
    
    # Assuming your dataframe is named 'to_plot'
    plot_multiindex_df(to_plot, name)

