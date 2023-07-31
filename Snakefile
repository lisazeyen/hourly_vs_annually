configfile: "config.yaml"

wildcard_constraints:
    policy="[\-a-zA-Z0-9\.]+"


RDIR = os.path.join(config['results_dir'], config['run'])
CDIR = config['costs_dir']


rule make_summary_all_networks:
    input:
        expand(RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/_{res_share}_{offtake_volume}volume_{storage}_summary.csv", **config['scenario'])


rule summarise_all_networks:
    input:
        expand(RDIR + "/summaries/{participation}/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}.yaml", **config['scenario'])


rule solve_all_networks:
    input:
        expand(RDIR + "/networks/{participation}/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}.nc", **config['scenario'])

rule plot_summary_all:
    input:
        expand(RDIR + "/plots/{participation}/{year}/{zone}/{palette}/used.pdf", **config["scenario"])

rule summarise_all_offtake:
    input:
        expand(RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/emissions.csv", **config["scenario"])
rule merge_plots:
    input:
        used=RDIR + "/plots/{participation}/{year}/{zone}/{palette}/used.pdf",
        config=RDIR + '/configs/config.yaml'
    output:
        final=RDIR + "/plots/{participation}/{year}/{zone}/{palette}/SUMMARY.pdf"
    threads: 2
    resources: mem_mb=2000
    script:
        'scripts/merge_plots.py'


rule plot_summary:
    input:
        summary=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/summary.csv",
        config=RDIR + '/configs/config.yaml'
    output:
        used=RDIR + "/plots/{participation}/{year}/{zone}/{palette}/used.pdf"
    threads: 2
    resources: mem_mb=2000
    script:
        'scripts/plot_summary.py'


rule make_summary:
    input:
        expand(RDIR + "/networks/{{participation}}/{{year}}/{{zone}}/{{palette}}/{policy}_{res_share}_{offtake_volume}volume_{storage}.nc",
               policy=config["scenario"]["policy"], res_share=config["scenario"]["res_share"],
               offtake_volume=config["scenario"]["offtake_volume"],storage=config["scenario"]["storage"])
    output:
        summary=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/_{res_share}_{offtake_volume}volume_{storage}_summary.csv"
    threads: 2
    resources: mem_mb=2000
    script: 'scripts/make_summary.py'


rule solve_base_network:
    input:
        config = RDIR + '/configs/config.yaml',
        network2030 = config['n_2030'],
        network2025 = config['n_2025'],
        costs2030=CDIR + "/costs_2030.csv",
        costs2025=CDIR + "/costs_2025.csv"
    output:
        network=RDIR + "/base/{participation}/{year}/{zone}/{palette}/base_{res_share}.nc"
    log:
        solver=RDIR + "/logs/{participation}/{year}/{zone}/{palette}/base_{res_share}_solver.log",
        python=RDIR + "/logs/{participation}/{year}/{zone}/{palette}/base_{res_share}_python.log",
        memory=RDIR + "/logs/{participation}/{year}/{zone}/{palette}/base_{res_share}_memory.log"
    threads: 12
    resources: mem_mb=8000
    script: "scripts/solve_network.py"


rule resolve_network:
    input:
        base_network=RDIR + "/base/{participation}/{year}/{zone}/{palette}/base_{res_share}.nc"
    output:
        network=RDIR + "/networks/{participation}/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}.nc"
    log:
        solver=RDIR + "/logs/{participation}/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_solver.log",
        python=RDIR + "/logs/{participation}/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_python.log",
        memory=RDIR + "/logs/{participation}/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}_memory.log"
    threads: 12
    resources: mem_mb=30000
    script: "scripts/resolve_network.py"

rule summarise_offtake:
    input:
        networks=expand(RDIR + "/networks/{{participation}}/{{year}}/{{zone}}/{{palette}}/{policy}_{res_share}_{offtake_volume}volume_{storage}.nc",
        policy=config["scenario"]["policy"], res_share=config["scenario"]["res_share"],
        offtake_volume=config["scenario"]["offtake_volume"],storage=config["scenario"]["storage"])
    output:
        csvs_emissions=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/emissions.csv",
        csvs_cf=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/cf.csv",
        csvs_supply_energy=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/supply_energy.csv",
        csvs_nodal_supply_energy=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/nodal_supply_energy.csv",
        csvs_nodal_capacities=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/nodal_capacities.csv",
        csvs_weighted_prices=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/weighted_prices.csv",
        csvs_curtailment=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/curtailment.csv",
        csvs_costs=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/costs.csv",
        csvs_nodal_costs=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/nodal_costs.csv",
        csvs_h2_costs=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/h2_costs.csv",
        csvs_emission_rate=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/emission_rate.csv",
        csvs_h2_gen_mix=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/h2_gen_mix.csv",
        csvs_attr_emissions=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/attr_emissions.csv",
        cf_plot = RDIR + "/graphs/{participation}/{year}/{zone}/{palette}/cf_electrolysis.pdf",
    threads: 2
    resources: mem=2000
    script: "scripts/summarise_offtake.py"

rule summarise_network:
    input:
        network=RDIR + "/networks/{participation}/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}.nc"
    output:
        yaml=RDIR + "/summaries/{participation}/{year}/{zone}/{palette}/{policy}_{res_share}_{offtake_volume}volume_{storage}.yaml"
    threads: 2
    resources: mem_mb=2000
    script: 'scripts/summarise_network.py'


rule copy_config:
    output: RDIR + '/configs/config.yaml'
    threads: 1
    resources: mem_mb=1000
    script: "scripts/copy_config.py"



# illustrate workflow
rule dag:
     message: "Plot dependency graph of the workflow."
     output:
         dot="workflow/dag.dot",
         graph="workflow/graph.dot",
         pdf="workflow/graph.pdf"
     shell:
         """
         snakemake --rulegraph > {output.dot}
         sed -e '1,2d' < {output.dot} > {output.graph}
         dot -Tpdf -o {output.pdf} {output.graph}
         """
