import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import boxcox
from src.mlProject import logger
from src.mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    @staticmethod
    def price_conversion(price_column: pd.Series) -> pd.Series:
        price_split = price_column.str.split()
        value = price_split.str[0].astype(float)
        unit = price_split.str[1].str.lower()

        value_converted = value.where(unit != "lac", value / 100)
        value_converted = value_converted.where(unit != "lakh", value_converted / 100)
        value_converted = value_converted.where(unit != "arab", value_converted * 100)

        return value_converted

    @staticmethod
    def convert_to_marla(area_series: pd.Series) -> pd.Series:
        area_split = area_series.str.split()
        value = area_split.str[0].astype(float)
        unit = area_split.str[1].str.lower()

        conversion_factors = {
            'kanal': 20,
            'marla': 1,
            'sqyd': 0.03,
            'sqft': 0.0033
        }

        value_converted = value * unit.map(conversion_factors)

        if value_converted.isna().any():
            raise ValueError(f"Unknown unit found in area_series: {area_series[value_converted.isna()].unique()}")

        return value_converted

    @staticmethod
    def skewness_removal(data: pd.DataFrame, column: str) -> pd.DataFrame:
        skewness = data[column].skew()
        print(f"The column '{column}' has a skewness of {skewness}")

        if skewness > 0.5:
            if (data[column] <= 0).any():
                raise ValueError(f"Column '{column}' contains non-positive values, cannot apply log transformation.")

            data[column] = np.log1p(data[column])
            skewness = data[column].skew()
            print(f"The skewness after log transformation is {skewness}")

            if skewness > 0.5:
                data[column] = np.sqrt(data[column])
                skewness = data[column].skew()
                print(f"The skewness after square root transformation is {skewness}")

                if skewness > 0.5:
                    data[column] = np.cbrt(data[column])
                    skewness = data[column].skew()
                    print(f"The skewness after cube root transformation is {skewness}")

                    if skewness > 0.5:
                        data[column], _ = boxcox(data[column] + 1)
                        skewness = data[column].skew()
                        print(f"The skewness after Box-Cox transformation is {skewness}")
        else:
            print(f"No transformation applied to '{column}' as skewness is not greater than 0.5")

        return data

    @staticmethod
    def cap_outliers(data: pd.DataFrame, column: str, threshold: float = 3) -> pd.DataFrame:
        mean = data[column].mean()
        std = data[column].std()

        upper_limit = mean + threshold * std
        lower_limit = mean - threshold * std

        data[column] = np.where(
            data[column] > upper_limit,
            upper_limit,
            np.where(
                data[column] < lower_limit,
                lower_limit,
                data[column]
            )
        )

        return data

    @staticmethod
    def binary_encoding(columns: list, data: pd.DataFrame) -> pd.DataFrame:
        ohe = OneHotEncoder(sparse_output=False)

        for column in columns:
            encoded_cols = ohe.fit_transform(data[[column]])
            encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out([column]), index=data.index)
            data = pd.concat([data, encoded_df], axis=1)
            data.drop(column, axis=1, inplace=True)

        return data

    def model_transformation(self):
        data = pd.read_csv(self.config.data_path)

        data["city"] = data["Address"].str.split().str[-1]

        data['Price'] = self.price_conversion(data['Price'])

        data['Area'] = self.convert_to_marla(data['Area'])

        data["Bedrooms"] = data["Bedrooms"].replace("10+", 10).astype(int)
        data['Bathrooms'] = data['Bathrooms'].replace('Studio', np.nan)
        data.dropna(subset=['Bathrooms'], inplace=True)
        data["Bathrooms"] = data["Bathrooms"].replace("10+", 10).astype(int)

        data = self.skewness_removal(data, "Price")

        data = self.cap_outliers(data, "Price")
        data = self.cap_outliers(data, "Area")

        # Extract the first two words from 'Address' to create a 'Town' column
        data["Town"] = data["Address"].str.split().str[:2].str.join(' - ')

        mean_price = data.groupby('Town')['Price'].mean()
        data['Town'] = data['Town'].map(mean_price)

        data = self.binary_encoding(columns=["Property Type", "city"], data=data)

        x = data.drop(columns=["Address", "Price"], axis=1)
        y = data["Price"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        std = StandardScaler()
        x_train_scaled = std.fit_transform(x_train)
        x_test_scaled = std.transform(x_test)

        return x_train_scaled, x_test_scaled, y_train, y_test

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        train, test = train_test_split(data, test_size=0.25, random_state=42)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Split data into training and test sets")
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")

        print(train.shape)
        print(test.shape)
