import pandas as pd
import os
from src.mlProject import logger
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from src.mlProject.entity.config_entity import ModelTrainingConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop(columns=[self.config.target_column])
        test_x = test_data.drop(columns=[self.config.target_column])
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        gbr = GradientBoostingRegressor(
            learning_rate=self.config.learning_rate, 
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            min_samples_split=self.config.min_samples_split,
            n_estimators=self.config.n_estimators
        )
        gbr.fit(train_x, train_y)

        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(gbr, model_path)
        logger.info(f"Model saved at {model_path}")