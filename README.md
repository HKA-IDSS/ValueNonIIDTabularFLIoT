# Flower_Experimentation_Framework

Experimentation setup for the paper *On the Utility of Non-i.i.d. Tabular IoT Data for Federated Learning*. 
The setup works by encapsulating the [Flower Federated Learning framework](https://flower.ai/), and extending it with functionality
for simulation of different scenarios and data quality measurements. We also provide the results obtained in our
experiments within the repository.

## Requirements

In order to use the framework, the following tools are needed:
- [anaconda/miniconda](https://www.anaconda.com/) (python can also be used, but the instructions 
will be provided only for anaconda)
- docker (support for docker-compose needed)

## Installation

Installation instructions are followed assuming you have installed anaconda.

1. (Optional, but preferable) Generate a new environment for anaconda, with Python 3.10 as the version. 

`conda create -n value-distance-noniid --python=3.10`

2. Start the environment. Do this everytime you want to work with the framework.

`conda activate value-distance-noniid`

3. Install all needed packages

`pip install -r requirements.txt`

4. Then, start the docker compose file for [Optuna](https://optuna.org/). This is needed, as the first time every configuration is run, 
it requires to run the optimization process.

`docker-compose up --build -d`

## Data

The data used for the study can be found here: [link](https://bwsyncandshare.kit.edu/s/rqqEJ9mn7THacRp).
Please, extract the folder "data" in the main folder.

## Run Experiment

In order to run the experiments in the platform, we provide with their configurations already created.
Only a few additional steps are needed to run the experiments, in the case of the manual partitions:

### Data Partitions

While Dirichlet partitions are automatically created by the training script, this is not the case for manual partitions.
In order to create a manual partition, run the following script from the main folder, replacing {name of partition} by
the actual python script that can be found in any of the folder in util/manual_partition_scripts:

`python util/manual_partition_scripts/{name of partition}.py`

### Running experiments

1. (Optional) Create a new experiment configuration. The possible parameters of the 
experiment configuration can be found in the file [Complete Experiment](https://github.com/JsAntoPe/Flower_Experimentation_Framework/blob/main/experiment_configuration/CompleteExperiment.yaml), or using any of the already existing yaml
files depicting existing experiments.
2. Run "python Main.py experiment_config_name"
    - Example: `python Main.py ExperimentFedAvgWineMLP100`

## Results

Results can be found in the folder results. Results of FL training experiments are in the folder _dataframes_, and
computations of the measures are in the folder _distances_values_.

To find a result in the folder _dataframes_, follow the directory structure with the following parameters in order:
1. Aggregation type
2. Name of dataset
3. Type of partition (whether manual or dirichlet)
4. Additional Parameter (for dirichlet is alpha value, for manual is name of partition)
5. Model

Results of the training for each metric are in Evaluation, 
and Shapley_Values contains the SV computation results for each metric.

To visualize training results, we recommend the use of a notebook. We have included all necessary notebooks to check
the results contained in the paper. The results of the paper can be checked in notebook _TransferLearning.ipynb_.

## Known and possible errors

- If an error about a non-existing record appears, this is because optuna could not find the study with the name of
the experiment. This is because either the FL training process was not executed, or the centralized training in the
notebook _GridSearchOptuna.ipynb_.

For any new errors, please raise an issue in GitHub, and we will get in contact as soon as possible.