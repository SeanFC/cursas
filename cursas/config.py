""" Read in and sort configuration options for Cursas """

# Standard library imports
import os
import configparser

class CursasConfig():
    """
    Read in, interpret and set defaults for Cursas configuration options
    """
    def __init__(self, config_file_path='config/cursas.ini'):
        """
        Constructor

        :param config_file_path: The path to the configuration file.
        If None then no settings are read in and all the defaults are used (Default config/cursas.ini).
        :type config_file_path: String
        """
        self.config_file_path = config_file_path

        # Set defaults for the given parameters
        ## File path settings
        self.data_dir = 'out/'
        self.sample_response_file_name = self.data_dir + 'sample_response.csv'
        self.run_ids_file_name = self.data_dir + 'run_ids.csv'
        self.full_table_file_name = self.data_dir + 'full_table.csv'

        ## Data pull settings
        self.user_agent = None
        self.external_website_db = ''

        # Pull in the settings from the given file.
        self.pull_in_settings(self.read_config_file())

    def read_config_file(self):
        """
        Read a configuration file into a ConfigParser object.

        :return: All of the configuration options from the given file.
        :rtype: configparser.ConfigParser
        """
        config = configparser.ConfigParser()
        if self.config_file_path:
            config.read(self.config_file_path)

        return config

    def pull_in_settings(self, config):
        """
        Pull in all the settings from a ConfigParser object.

        :param config: All of the configuration options to set
        :type config: configparser.ConfigParser
        """

        if 'database' in config:
            self.data_dir = os.path.expanduser(config['database'].get('data_dir', self.data_dir))
            self.run_ids_file_name = self.data_dir + config['database'].get('run_ids_file_name', self.run_ids_file_name)
            self.full_table_file_name = self.data_dir + config['database'].get('full_table_file_name', self.full_table_file_name)
            self.sample_response_file_name = self.data_dir + config['database'].get(
                    'sample_response_file_name',
                    self.sample_response_file_name
                    )

        if 'pull' in config:
            self.user_agent = config['pull'].get('user_agent', self.user_agent)
            self.external_website_db = config['pull'].get('external_website_db', self.external_website_db)
