#TODO: This module makes loads of globals. It should return and object that can be used for config or something like that

import os
import configparser

config_file_path='config/cursas.ini'

# Read in any configuration file given
config = configparser.ConfigParser()
if config_file_path:
    config.read(config_file_path)

data_dir = os.path.expanduser(config['database']['data_dir'])
sample_response_file_name = data_dir + config['database']['sample_response_file_name']
run_ids_file_name = data_dir + config['database']['run_ids_file_name']
full_table_file_name = data_dir + config['database']['full_table_file_name']
