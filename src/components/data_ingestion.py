import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logger
from src.utils import read_yaml, load_data
from pathlib import Path


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data: handle missing values, duplicates, and empty strings."""
    try:
        initial_shape = df.shape
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        if 'clean_comment' in df.columns:
            df = df[df['clean_comment'].str.strip() != '']

        logger.info(f"Preprocessing completed: Original shape {initial_shape} -> Final shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise CustomException(e, sys)


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save train and test datasets inside data/raw directory."""
    try:
        data_path = Path(data_path)
        os.makedirs(data_path, exist_ok=True)

        train_path = os.path.join(data_path, "train.csv")
        test_path = os.path.join(data_path, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.info(f"Train data saved to {train_path} | Shape: {train_data.shape}")
        logger.info(f"Test data saved to {test_path} | Shape: {test_data.shape}")

    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise CustomException(e, sys)


def initiate_data_ingestion() -> None:
    """Run the entire data ingestion pipeline."""
    try:
        params = read_yaml(Path("params.yaml"))
        test_size = params.data_ingestion.test_size

        df = load_data(params.data_ingestion.url)
        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=42
        )

        save_data(train_data, test_data, params.data_ingestion.data_path)
        logger.info("Data ingestion pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Error in data ingestion pipeline: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    initiate_data_ingestion()
