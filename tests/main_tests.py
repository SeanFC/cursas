import unittest
import os

from cursas.config import CursasConfig

class TestConfig(unittest.TestCase):
    pull_tmp_file_path = '/tmp/test_cursas_no_pull_config.ini'
    db_tmp_file_path = '/tmp/test_cursas_no_db_config.ini'

    def setUp(self):
        with open(self.pull_tmp_file_path, "w") as config_file:
            config_file.write('''
            [database]
            run_path=test
            ''')

        with open(self.db_tmp_file_path, "w") as config_file:
            config_file.write('''
            [pull]
            external_website_db=www.example.com
            ''')


    def test_no_pull_section(self):
        self.assertIsInstance(CursasConfig(self.pull_tmp_file_path).external_website_db, str)

    def test_no_database_section(self):
        self.assertIsInstance(CursasConfig(self.db_tmp_file_path).data_dir, str)

    def tearDown(self):
        os.remove(self.pull_tmp_file_path)

if __name__ == '__main__':
    unittest.main()

