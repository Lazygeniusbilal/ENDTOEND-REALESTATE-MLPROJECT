import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath('__file__')), '..', 'src')))

# Import constants
from src.mlProject.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.mlProject.utils.common import read_yaml, create_directories
from src.mlProject import logger
from src.mlProject.entity.config_entity import (DataIngestionConfig)

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH, schema_filepath=SCHEMA_FILE_PATH):
        self.config_filepath = config_filepath
        self.params_filepath = params_filepath
        self.schema_filepath = schema_filepath

        logger.info(f"Using config file at: {self.config_filepath}")
        logger.info(f"Using params file at: {self.params_filepath}")
        logger.info(f"Using schema file at: {self.schema_filepath}")

        self.config = read_yaml(self.config_filepath)
        self.params = read_yaml(self.params_filepath)
        self.schema = read_yaml(self.schema_filepath)
        
        logger.info(f"Configuration loaded: {self.config}")

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.get('data_ingestion', {})
        create_directories([config.get('root_dir', 'artifacts/data_ingestion')])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.get('root_dir', 'artifacts/data_ingestion'),
            source_URL=config.get('source_URL', ''),
            local_data_file=config.get('local_data_file', ''),
            unzip_dir=config.get('unzip_dir', '')
        )
        
        return data_ingestion_config
