import json
import pickle
import sys

import numpy as np

import clban as clb

# Read the configuration file
config_file = sys.argv[1]
with open(config_file, "r") as c_file:
    config = json.load(c_file)


# Set the seed
if "seed" in config:
    np.random.seed(config["seed"])
else:
    config["seed"] = None


# Prepare the algorithm
alg = None
alg_config = config["algorithm"]
if alg_config["algorithm"] == "SplitItemsH":
    alg = clb.SplitItemsH(num_items=alg_config["num_items"], delta=alg_config["delta"])
elif alg_config["algorithm"] == "SplitItemsS":
    alg = clb.SplitItemsS(
        num_items=alg_config["num_items"],
        delta=alg_config["delta"],
        max_search_iters=alg_config["max_search_iters"],
    )
elif alg_config["algorithm"] == "QBClusterH":
    alg = clb.QBClusterH(
        num_items=alg_config["num_items"],
        num_clusters=alg_config["num_clusters"],
        delta=alg_config["delta"],
    )
elif alg_config["algorithm"] == "QBClusterS":
    alg = clb.QBClusterS(
        num_items=alg_config["num_items"],
        num_clusters=alg_config["num_clusters"],
        delta=alg_config["delta"],
        max_search_iters=alg_config["max_search_iters"],
    )

if alg is None:
    supported_alg = ["SplitItemsH", "SplitItemsS", "QBClusterH", "QBClusterS"]
    raise ValueError(
        f'Algorithm {alg_config["algorithm"]} not supported. Use one of {supported_alg}'
    )


# Prepare the oracle
oracle = None
oracle_config = config["oracle"]
if oracle_config["oracle"] == "PPOracle":
    oracle = clb.PPOracle(
        num_items=oracle_config["num_items"],
        clusters=np.asarray(oracle_config["clusters"]),
        p=oracle_config["p"],
        q=oracle_config["q"],
    )
elif oracle_config["oracle"] == "PerturbedPPOracle":
    oracle = clb.PerturbedPPOracle(
        num_items=oracle_config["num_items"],
        clusters=np.asarray(oracle_config["clusters"]),
        p=oracle_config["p"],
        q=oracle_config["q"],
        perturbation_noise=oracle_config["perturbation_noise"],
    )


if oracle is None:
    supported_oracles = ["PPOracle", "PerturbedPPOracle"]
    raise f'Oracle {oracle_config["oracle"]} not supported. Use one of {supported_oracles}'


# Run the experiment
experiment = clb.Experiment(oracle, alg)
results = experiment.run()


# Write the output
out_file = config["out_file"]
with open(out_file, "wb") as out_f:
    output = dict()
    output["seed"] = config["seed"]
    output["exp_config"] = experiment.config()
    output["results"] = results
    pickle.dump(output, out_f)
