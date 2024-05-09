import pandas as pd

# a dictionary of industry sector symbols
group_dictionary = pd.read_csv('data/group_dictionary.csv', sep=';')
column_names_dictionary = pd.read_csv('data/column_names_dictionary.csv', sep=';')
test_data_no_target = pd.read_csv('data/test_data_no_target.csv', sep=';', decimal=',')
training_data = pd.read_csv('data/training_data.csv', sep=';', decimal=',')
