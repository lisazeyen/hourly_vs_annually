configfile: "config.yaml"

wildcard_constraints:
    policy="[\-a-zA-Z0-9\.]+"


RDIR = os.path.join(config['results_dir'], config['run'])
CDIR = config['costs_dir']


rule solve_all_networks:
    input:
        expand(RDIR + "/networks/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}.nc", **config['scenario'])


rule summarise_all_offtake:
    input:
        expand(RDIR + "/csvs/{year}/{zone}/{palette}/emissions.csv", **config["scenario"])
rule merge_plots:
    input:
        used=RDIR + "/plots/{year}/{zone}/{palette}/used.pdf",
        config=RDIR + '/configs/config.yaml'
    output:
        final=RDIR + "/plots/{year}/{zone}/{palette}/SUMMARY.pdf"
    threads: 2
    resources: mem_mb=2000
    script:
        'scripts/merge_plots.py'


rule solve_base_network:
    input:
        config = RDIR + '/configs/config.yaml',
        network2030 = config['n_2030'],
        network2025 = config['n_2025'],
        costs2030=CDIR + "/costs_2030.csv",
        costs2025=CDIR + "/costs_2025.csv"
    output:
        network=RDIR + "/base/{year}/{zone}/{palette}/base_{res_share}_{offtake_volume}volume.nc"
    log:
        solver=RDIR + "/logs/{year}/{zone}/{palette}/base_{res_share}_{offtake_volume}volume_solver.log",
        python=RDIR + "/logs/{year}/{zone}/{palette}/base_{res_share}_{offtake_volume}volume_python.log",
        memory=RDIR + "/logs/{year}/{zone}/{palette}/base_{res_share}_{offtake_volume}volume_memory.log"
    threads: 12
    resources: mem_mb=8000
    script: "scripts/solve_network.py"


rule resolve_network:
    input:
        base_network=RDIR + "/base/{year}/{zone}/{palette}/base_{res_share}_{offtake_volume}volume.nc"
    output:
        network=RDIR + "/networks/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}.nc"
    log:
        solver=RDIR + "/logs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_solver.log",
        python=RDIR + "/logs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_python.log",
        memory=RDIR + "/logs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_memory.log"
    threads: 12
    resources: mem_mb=30000
    script: "scripts/resolve_network.py"

rule summarise_offtake:
    input:
        networks=expand(RDIR + "/networks/{{participation}}/{{year}}/{{zone}}/{{palette}}/{policy}_{res_share}_{offtake_volume}volume_{storage}.nc",
        policy=config["scenario"]["policy"], res_share=config["scenario"]["res_share"],
        offtake_volume=config["scenario"]["offtake_volume"],storage=config["scenario"]["storage"])
    output:
        csvs_emissions=RDIR + "/csvs/{year}/{zone}/{palette}/emissions.csv",
        csvs_cf=RDIR + "/csvs/{year}/{zone}/{palette}/cf.csv",
        csvs_supply_energy=RDIR + "/csvs/{year}/{zone}/{palette}/supply_energy.csv",
        csvs_nodal_supply_energy=RDIR + "/csvs/{year}/{zone}/{palette}/nodal_supply_energy.csv",
        csvs_nodal_capacities=RDIR + "/csvs/{year}/{zone}/{palette}/nodal_capacities.csv",
        csvs_weighted_prices=RDIR + "/csvs/{year}/{zone}/{palette}/weighted_prices.csv",
        csvs_curtailment=RDIR + "/csvs/{year}/{zone}/{palette}/curtailment.csv",
        csvs_costs=RDIR + "/csvs/{year}/{zone}/{palette}/costs.csv",
        csvs_nodal_costs=RDIR + "/csvs/{year}/{zone}/{palette}/nodal_costs.csv",
        csvs_h2_costs=RDIR + "/csvs/{year}/{zone}/{palette}/h2_costs.csv",
        csvs_emission_rate=RDIR + "/csvs/{year}/{zone}/{palette}/emission_rate.csv",
        csvs_h2_gen_mix=RDIR + "/csvs/{year}/{zone}/{palette}/h2_gen_mix.csv",
        csvs_attr_emissions=RDIR + "/csvs/{year}/{zone}/{palette}/attr_emissions.csv",
        csvs_price_duration=RDIR + "/csvs/{year}/{zone}/{palette}/price_duration.csv",
        # cf_plot = RDIR + "/graphs/{year}/{zone}/{palette}/cf_electrolysis.pdf",

    threads: 2
    resources: mem=2000
    script: "scripts/summarise_offtake.py"


rule copy_config:
    output: RDIR + '/configs/config.yaml'
    threads: 1
    resources: mem_mb=1000
    script: "scripts/copy_config.py"
