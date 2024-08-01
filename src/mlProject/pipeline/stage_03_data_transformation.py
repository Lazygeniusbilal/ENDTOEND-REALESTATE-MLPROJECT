from src.mlProject.config.configuration import ConfigurationManager
from pathlib import Path
from src.mlProject.components.data_transformation import DataTransformation
from src.mlProject import logger

STAGE_NAME = "Data_Transformation"



class DataTransformationPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager(
        config_filepath=Path("config/config.yaml"),
        params_filepath=Path("params.yaml"),
        schema_filepath=Path("schema.yaml")
    )
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        x_train_scaled, x_test_scaled, y_train, y_test = data_transformation.model_transformation()
        data_transformation.train_test_spliting()

# Main script
if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e