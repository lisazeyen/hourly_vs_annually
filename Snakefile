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
        expand(RDIR + "/csvs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_emissions.csv", **config["scenario"])

rule plot_all:
    input:
        expand(RDIR + "/graphs/{year}/{zone}/{palette}/cf_electrolysis.pdf", **config["scenario"])

if config["solving_option"] == "twostep":
    rule solve_base_network:
        input:
            config = RDIR + '/configs/config.yaml',
            network2030 = config['n_2030'],
            network2025 = config['n_2025'],
            costs2030=CDIR + "/costs_2030.csv",
            costs2025=CDIR + "/costs_2025.csv"
        output:
            prenetwork=RDIR + "/base/{year}/{zone}/{palette}/prebase_{res_share}_{offtake_volume}volume.nc",
            network=RDIR + "/base/{year}/{zone}/{palette}/base_{res_share}_{offtake_volume}volume.nc"
        log:
            solver=RDIR + "/logs/{year}/{zone}/{palette}/base_{res_share}_{offtake_volume}volume_solver.log",
            python=RDIR + "/logs/{year}/{zone}/{palette}/base_{res_share}_{offtake_volume}volume_python.log",
            memory=RDIR + "/logs/{year}/{zone}/{palette}/base_{res_share}_{offtake_volume}volume_memory.log"
        threads: 12
        resources: mem_mb=80000
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
        resources: mem_mb=62500
        script: "scripts/resolve_network.py"

if config["solving_option"] == "together":
    rule solve_network_together:
        input:
            config = RDIR + '/configs/config.yaml',
            network2030 = config['n_2030'],
            network2025 = config['n_2025'],
            costs2030=CDIR + "/costs_2030.csv",
            costs2025=CDIR + "/costs_2025.csv"
        output:
            network=RDIR + "/networks/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}.nc"
        log:
            solver=RDIR + "/logs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_solver.log",
            python=RDIR + "/logs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_python.log",
            memory=RDIR + "/logs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_memory.log"
        threads: 12
        resources: mem_mb=80000
        script: "scripts/solve_network_together.py"


rule summarise_offtake:
    input:
        network=RDIR + "/networks/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}.nc",
    output:
        csvs_emissions=RDIR + "/csvs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_emissions.csv",
        csvs_cf=RDIR + "/csvs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_cf.csv",
        csvs_supply_energy=RDIR + "/csvs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_supply_energy.csv",
        csvs_nodal_supply_energy=RDIR + "/csvs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_nodal_supply_energy.csv",
        csvs_nodal_capacities=RDIR + "/csvs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_nodal_capacities.csv",
        csvs_weighted_prices=RDIR + "/csvs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_weighted_prices.csv",
        csvs_curtailment=RDIR + "/csvs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_curtailment.csv",
        csvs_costs=RDIR + "/csvs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_costs.csv",
        csvs_nodal_costs=RDIR + "/csvs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_nodal_costs.csv",
        csvs_h2_costs=RDIR + "/csvs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_h2_costs.csv",
        csvs_emission_rate=RDIR + "/csvs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_emission_rate.csv",
        csvs_h2_gen_mix=RDIR + "/csvs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_h2_gen_mix.csv",
        csvs_attr_emissions=RDIR + "/csvs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_attr_emissions.csv",
        csvs_price_duration=RDIR + "/csvs/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_price_duration.csv",
    threads: 2
    resources: mem=2000
    script: "scripts/summarise_offtake.py"

rule plot_offtake:
    input:
        csvs=expand(RDIR + "/csvs/{{year}}/{{zone}}/{{palette}}/{policy}_{res_share}_{offtake_volume}volume_{storage}_emissions.csv",
        policy=config["scenario"]["policy"], res_share=config["scenario"]["res_share"],
        offtake_volume=config["scenario"]["offtake_volume"],storage=config["scenario"]["storage"])
    output:
        cf_plot = RDIR + "/graphs/{year}/{zone}/{palette}/cf_electrolysis.pdf",

    threads: 2
    resources: mem=2000
    script: "scripts/plot_offtake.py"


rule copy_config:
    output: RDIR + '/configs/config.yaml'
    threads: 1
    resources: mem_mb=1000
    script: "scripts/copy_config.py"
