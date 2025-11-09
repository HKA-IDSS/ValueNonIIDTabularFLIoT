import optuna
import pymysql

# Optuna url for connection within the docker-compose cluster.
# optuna_studies_url = "mysql+pymysql://root:example@db:3306/optuna_studies"

# Optuna url for connection locally.
optuna_studies_url = "mysql+pymysql://root:dg-connection@localhost:3306/optuna_studies"


def optuna_create_study(name, direction):
    conn = pymysql.connect(host='localhost',
                           user='root',
                           password='dg-connection')
    # conn.cursor().execute("drop database if exists {};".format(name))
    conn.cursor().execute("create database if not exists optuna_studies")
    try:
        study = optuna.create_study(
            storage=optuna_studies_url,  # Specify the storage
            # URL here.
            study_name=name,
            directions=direction,
            load_if_exists=False
        )
    except optuna.exceptions.DuplicatedStudyError:
        optuna.delete_study(study_name=name,
                            storage=optuna_studies_url)
        study = optuna.create_study(
            storage=optuna_studies_url,  # Specify the storage
            # URL here.
            study_name=name,
            directions=direction
        )
    return study


def load_study(name_study):
    study = optuna.load_study(
        study_name=name_study,
        storage=optuna_studies_url
    )
    return study

# Command to run:
# optuna-dashboard mysql+pymysql://root:example@localhost:3306/optuna_studies
