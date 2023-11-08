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
    for (country, policy, storage) in df.index:
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
    
    
  
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull

# New color dictionary for storage types
storage_colors = {
    'flexibledemand': 'green',
    'mtank': 'orange',
    'nostore': 'red'
}

# New marker dictionary for countries
country_markers = {
    "DE-2025": "p",
    "DE-2030": "8",
    "ES": "4",
    "PT": "s",
    "CZ": 'o',
    "NL": '*',
    "PL": "^",
}

policy_colors = {
    'grd': 'red',  # Clear red color for "grid"
    'offgrid': '#332288',  # Indigo shade
    'monthly': '#117733',  # Dark green
    'res1p0': '#DDCC77',  # Khaki
    'res1p2': '#88CCEE',  # Sky blue
    'res1p3': '#CC6677',  # Pinkish
    "exl1p2": '#AA4499',  # Purple
    "exl1p3": '#44AA99',  # Teal
}


# Update rename_scenarios with the full policy list
rename_scenarios.update(policy_markers)


from matplotlib.patches import Polygon

def plot_multiindex_df_hull(dfs, name):
    
    cf = dfs.cf
    df = dfs[["h2_cost", "emissions"]]
    # Get unique values for countries, policies, and storage types
    countries = df.index.get_level_values(0).unique()
    policies = df.index.get_level_values(1).unique()
    storages = df.index.get_level_values(2).unique()
    
    # Color map for countries
    # Colorblind-friendly color palette
    cb_palette = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', 
                  '#984ea3', '#999999', '#e41a1c', '#dede00']
    

    # Initialize dictionary to hold policy points
    policy_points = {policy: [] for policy in policies}



    plots = ["all", "option1", "option2", "option3"]
    opts_dict = {"all": "",
                 "option1": "hourly matching with low cost storage",
                 "option2": "capacity factors of electrolysis < 70%",
                 "option3": "largely decarbonised background system (RES share >= 80%)"}
    
    for plot in plots:
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(opts_dict[plot], fontsize=16, fontweight='bold')
    
        # Collect data points for each policy
        for (country, policy, storage) in df.index:
            # Select data for each line
            data = df.loc[(country, policy, storage)]
            cf_i = cf.loc[(country, policy, storage)]
            point = (data['h2_cost'], data['emissions'])
            policy_points[policy].append(point)
            
            # make some markers less transparent
            option1 = (policy in ["exl1p2", "offgrid"]) and (storage in ["flexibledemand", "mtank"])
            option2 = (cf_i<=0.7) and (policy!="grd")
            option3 = (country in ["DE-2030"]) and (policy!="grd")
            if plot=="all":
                bool_o = option1 or option2 or option3
            elif plot=="option1":
                bool_o = option1
            elif plot=="option2":
                bool_o = option2
            elif plot=="option3":
                bool_o = option3
                
                
            if bool_o:
                alpha = 0.8
                edge_color = 'black'
                line_width = 1
                z_order = 3
            else:
                alpha = 0.5
                edge_color = 'none'
                line_width = 0
                z_order = 2
    
            # Plot each data point with its respective storage color and country marker
            ax.scatter(data['h2_cost'], data['emissions'], 
                color=storage_colors[storage], 
                edgecolors=edge_color, 
                linewidths=line_width,
                marker=country_markers.get(country, 'o'), 
                s=70,  # Marker size
                alpha=alpha, 
                zorder=z_order,
                label=f"{country}, {policy}, {storage}")
    
        # Add background color for each policy using fill_between
        for policy, points in policy_points.items():
            if points:  # Check if there are any points for the policy
                sorted_points = sorted(points)  # Sort points based on h2_cost (x-value)
                x_values, y_values = zip(*sorted_points)
                min_y = min(y_values)
                max_y = max(y_values)
                ax.fill_between(x_values, min_y-1, max_y+1,
                                color=policy_colors[policy],
                                alpha=0.2,
                                step='mid')
    
    
        
        # ... [Rest of your plotting code to create and set legends, labels, grid]
        # Set the labels for the axes
        ax.set_xlabel("cost \n [Euro/kg$_{H_2}$]")
        ax.set_ylabel("consequential emissions \n [kg$_{CO_2}$/kg$_{H_2}$]")
        # Set grid
        ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.7)
        
        # Create legend handles for policies using a patch for the convex hull color
        rename_scenarios = {"res1p0": "annually", "exl1p0": "hourly", "offgrid": "hourly",
                            "grd": "grid", "monthly": "monthly",
                            "res1p2": "annually excess 20%", "exl1p2": "hourly excess 20%",
                            "res1p3": "annually excess 30%", "exl1p3": "hourly excess 30%",}
    
        policy_handles = [mpatches.Patch(color=policy_colors[policy], label=rename_scenarios[policy], alpha=0.3)
                          for policy in wished_policies]
        
        # Create legend handles for storage types using the new color scheme
        storage_handles = [mlines.Line2D([], [], color=storage_colors[storage], marker='o', linestyle='None',
                                          markersize=10, label=storage)
                           for storage in storages]
        
        # Assuming country_markers is a dictionary like policy_markers, providing a specific marker for each country
        # Create legend handles for countries
        country_handles = [mlines.Line2D([], [], color='black', marker=country_markers[country], linestyle='None',
                                          markersize=10, label=country)
                           for country in countries]
        
        # Combine handles
        handles = country_handles + policy_handles + storage_handles
        
        # Add legend to the plot
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set x-axis limit to 22
        ax.set_xlim(left=0, right=22)  # Only set the upper bound to 22
        
        # Add a vertical dashed line at x=5 going up to y=0
        ax.axvline(x=5, color='#404040', linestyle='--', linewidth=1) # ymin=ax.get_ylim()[0], ymax=0
        
        # Add a horizontal dashed line at y=0 going right until x=5
        ax.axhline(y=0, color='#404040', linestyle='--', linewidth=1) # xmin=0, xmax=5/ax.get_xlim()[1],
    
        # Add text above the x-axis between x=0 and x=5 saying "low cost"
        ax.text(2.5, ax.get_ylim()[0], 'low cost', # x=2.5 centers the text between 0 and 5
                ha='center', # horizontal alignment is center
                va='bottom', # vertical alignment is bottom
                fontsize=10)
        
        # Add vertical text parallel to the y-axis saying "low emissions" below y<0
        ax.text( 0.1, ax.get_ylim()[0]/10, 'low emissions', # y=-5 places the text below y=0
                rotation='vertical', # rotate the text vertically
                ha='left', # horizontal alignment is left
                va='top', # vertical alignment is top (since it's rotated, "top" will be towards the x-axis)
                fontsize=10)
    
    
        plt.savefig(f"/home/lisa/Documents/hourly_vs_annually/results/summary_cost_emissions_{year}_{name}_hull_{plot}.pdf",
                    bbox_inches="tight")

        
#%%
compare_p = "offgrid"
emissions = {}
h2_cost = {}
cf = {}
volume = "3200"
country_dict = {"DE-2025": "gas_price_35_nocrossover",
                "DE-2030": "DE_NL",
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
    
    
    # capacity factor
    cf[ct] = pd.read_csv(path + "cf_together.csv",
                                    index_col=[0,1,2,3]).xs((float(res_share), float(volume)),
                                                         level=[1,2])
#%%
emissions_all = pd.concat(emissions, axis=1).droplevel(1, axis=1)
h2_cost_all = pd.concat(h2_cost, axis=1)
cf_all = pd.concat(cf, axis=1).droplevel(1, axis=1).unstack(level=0).unstack()

summed_cost = h2_cost_all.sum()
em = emissions_all.unstack(level=0).unstack()
# tot = pd.concat([summed_cost, em], axis=1)
tot = pd.concat([summed_cost, em, cf_all], axis=1)
tot.columns = ["h2_cost", "emissions", "cf"]
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
    plot_multiindex_df_hull(to_plot, name)

