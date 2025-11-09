""" Experimental 

    -experiment config generator-
    makes it easier to generate multiple configurations at the same time.

    TODO: Support more than one split set
    
    Input Example: "FedAvg,FedKD,FedGKD,FedLF", "iris,wine,har,adult", "0.01,0.1,1,10,100", "1", "20,20,20,20,20"
    multiple_config_generator.py --dataset "iris,wine,har,adult" --strategy "FedAvg,FedKD,FedGKD,FedLF" --alpha "0.01,0.1,1,10,100" --seed "1" --split "20,20,20,20,20" ----model "tabnet,mlp"
"""
import argparse

def make_parser():
    parser = argparse.ArgumentParser(description='Pipeline for federated learning with virtual clients')
    parser.add_argument('--dataset', action='store', help="List of datasets", required=True)
    parser.add_argument('--strategy', action='store', help="List of strategies", required=True)
    parser.add_argument('--split', action='store', help="Splits for clients", required=True)
    parser.add_argument('--alpha', action='store', help="List of alpha values", required=True)
    parser.add_argument('--seed', action='store', help="List of seeds", required=True)
    parser.add_argument('--model', action='store', help="TFModel", required=True)
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()

    dataset= args.dataset
    strategy = args.strategy
    splits = args.split
    alphas = args.alpha
    seed = args.seed
    m = args.model
    
    # Transform inputs to lists
    dataset = dataset.split(",")
    strategy = strategy.split(",")
    splits = [int(split) for split in splits.split(",")]
    alphas = alphas.split(",")
    seed = seed.split(",")
    models = m.split(",")

    
    # Create all possible combinations
    max_combis = len(dataset) * len(strategy) * len(seed) * len(alphas) * len(models)
    print(f"Generating {max_combis} experiment configurations.")
    exp_config_combinations = []

    exp_dict = {}
    for dset in dataset:
        for strat in strategy:
            for alpha in alphas:
                for se in seed:
                    for model in models:
                        exp_dict["strategy"] = strat
                        exp_dict["dataset"] = dset
                        exp_dict["alpha"] = alpha
                        exp_dict["seed"] = se
                        exp_dict["splits"] = splits
                        exp_dict["model"] = model
                        exp_config_combinations.append(exp_dict)
                        exp_dict = {}              
    
    # Create config files for each combination
    for exp_config in exp_config_combinations:
        client_log = open(f'GEN_Experiment{exp_config["strategy"]}{exp_config["dataset"].strip()}{exp_config["model"]}{exp_config["alpha"]}.yaml', 'a')
        client_log.write(f'strategy: {exp_config["strategy"]}\n')
        client_log.write(f'dataset: {exp_config["dataset"]}\n')
        client_log.write(f'model: {exp_config["model"]}\n')
        client_log.write(f'alpha: {exp_config["alpha"]}\n')
        client_log.write(f'seed: {exp_config["seed"]}\n')
        client_log.write(f'batch_size: 32\n')
        client_log.write(f'rounds: 20\n')
        client_log.write(f'data_split: {exp_config["splits"]}\n')
        client_log.close()
    
if __name__ == "__main__":
    main()    