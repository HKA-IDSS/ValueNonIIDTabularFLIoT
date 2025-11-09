import os

import numpy as np
from folktables import ACSDataSource, ACSIncome, generate_categories, folktables, adult_filter

from Definitions import ROOT_DIR, countries_selected

def acs_get_full_data(list_of_states, year='2018'):
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person',
                                root_dir=ROOT_DIR + os.sep + 'data' + os.sep + 'datasets' + os.sep + 'New_Adult')
    acs_data14 = data_source.get_data(states=list_of_states, download=True)
    definition_df = data_source.get_definitions(download=True)
    categories = generate_categories(features=ACSIncome.features, definition_df=definition_df)
    X, y, _ = ACSIncome.df_to_pandas(acs_data14, categories=categories, dummies=True)
    X = X.apply(lambda x: x.apply(lambda cell: 1 if cell else 0))

    return X, y

# if __name__ == "__main__":
#     data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person',
#                                 root_dir=ROOT_DIR + os.sep + 'data' + os.sep + 'datasets' + os.sep + 'New_Adult')
#     acs_data14 = data_source.get_data(states=countries_selected, download=True)
#     definition_df = data_source.get_definitions(download=True)
#
#     categories = generate_categories(features=ACSIncome.features, definition_df=definition_df)
#     X, y, _ = ACSIncome.df_to_pandas(acs_data14, categories=categories, dummies=True)
#     X = X.apply(lambda x: x.apply(lambda cell: 1 if cell else 0))
#
#     X.to_csv(ROOT_DIR + os.sep + "data" + os.sep + "datasets" + os.sep + "New_Adult" + os.sep + "X.csv")
#     y.to_csv(ROOT_DIR + os.sep + "data" + os.sep + "datasets" + os.sep + "New_Adult" + os.sep + "y.csv")
