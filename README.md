# Temporal regulation of renewable supply for electrolytic hydrogen

Temporal regulation of renewable supply for electrolytic hydrogen is an open model to model hydrogen production with different
regulation standards in a selected country in Europe.

This repository contains the code to reproduce the complete workflow behind the [manuscript](https://zenodo.org/records/8324521).

## Abstract

Electrolytic hydrogen produced using renewable electricity can help lower carbon dioxide emissions in sectors where feedstocks, reducing agents, dense fuels or high temperatures are required.
This study investigates the implications of various standards being proposed to certify that the grid electricity used is renewable. The standards vary in how strictly they match the renewable generation to the electrolyser demand in time and space. Using an energy system model, we compare electricity procurement strategies to meet a constant hydrogen demand for selected European countries in 2025 and 2030. We compare cases where no additional renewable generators are procured with cases where the electrolyser demand is matched to additional supply from local renewable generators on an annual, monthly or hourly basis. We show that local additionality is required to
guarantee low emissions. For the annually and monthly matched case, we demonstrate that baseload operation of the electrolysis leads to using fossil-fuelled generation from the grid for some hours, resulting in higher emissions than the case without hydrogen demand.
In the hourly matched case, hydrogen production does not increase system-level emissions, but baseload operation results in high costs for providing constant supply if only wind, solar and short-term battery storage are available.
Flexible operation or buffering hydrogen with storage, either in steel tanks or underground caverns, reduces the cost penalty of hourly versus annual matching to 7--8%. Hydrogen production with monthly matching can reduce system emissions if the electrolysers operate flexibly or the renewable generation share is large. The largest emission reduction is achieved with hourly matching when surplus electricity generation can be sold to the grid. We conclude that flexible operation of the electrolysis should be supported to guarantee low emissions and low hydrogen production costs.

## Installation

1. **Setup and Environment**:

   I generally recommend using `mamba` over `conda`. You can find the installation instructions for `mamba` [here](https://mamba.readthedocs.io/en/latest/installation.html).

   From your terminal, run the following commands:

   ```bash
   git clone https://github.com/lisazeyen/hourly_vs_annually.git
   mamba env create -f envs/environment.yaml
   conda activate hourly
   ```

2. **Running the Workflow**:

   To execute the workflow, use the following command:

   ```bash
   snakemake plot_all -j8
   ```

3. **Configuration**:

   You can modify assumptions in the `config` (e.g., country, storage types, demand volume, CO2 limits, etc.).

---
