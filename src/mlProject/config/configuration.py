import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath('__file__')), '..', 'src')))
from pathlib import Path

# Import constants
from src.mlProject.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.mlProject.utils.common import read_yaml, create_directories
from src.mlProject import logger
from src.mlProject.entity.config_entity import (DataIngestionConfig, 
                                                DataValidationConfig, 
                                                DataTransformationConfig, ModelTrainingConfig)

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


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config

class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = Path(CONFIG_FILE_PATH),
        params_filepath: Path = Path(PARAMS_FILE_PATH),
        schema_filepath: Path = Path(SCHEMA_FILE_PATH)
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config["artifacts_root"]])

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config["data_transformation"]

        create_directories([config["root_dir"]])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config["root_dir"]),
            data_path=Path(config["data_path"])
        )

        return data_transformation_config
    

class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = Path(CONFIG_FILE_PATH),
        params_filepath: Path = Path(PARAMS_FILE_PATH),
        schema_filepath: Path = Path(SCHEMA_FILE_PATH)
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_model_trainer_config(self) -> ModelTrainingConfig:
        config = self.config.model_trainer
        params = self.params.GradientBoostingRegressor

        # Ensure schema is accessed correctly
        schema = self.schema.to_dict() if isinstance(self.schema, ConfigBox) else self.schema
        target_column = schema.get('TARGET_COLUMN', {}).get('name', 'default_target_column')

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            learning_rate=params.learning_rate,
            max_depth=params.max_depth,
            min_samples_leaf=params.min_samples_leaf,
            min_samples_split=params.min_samples_split,
            n_estimators=params.n_estimators,
            target_column=target_column
        )

        return model_trainer_config