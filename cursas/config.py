import os
import configparser

class CursasConfig():
    def __init__(self, config_file_path):
        # Read in any configuration file given
        config = configparser.ConfigParser()
        if config_file_path:
            config.read(config_file_path)

        self.data_dir = os.path.expanduser(config['database']['data_dir'])
        self.sample_response_file_name = self.data_dir + config['database']['sample_response_file_name']
        self.run_ids_file_name = self.data_dir + config['database']['run_ids_file_name']
        self.full_table_file_name = self.data_dir + config['database']['full_table_file_name']

        self.user_agent = config['pull']['user_agent']
        self.external_website_db = config['pull']['external_website_db']


def get_config(config_file_path='config/cursas.ini'):
    return CursasConfig(config_file_path)
