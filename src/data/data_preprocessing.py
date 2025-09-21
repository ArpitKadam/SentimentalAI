import os
import re
import sys
import pandas as pd
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.logger import logger
from src.exception import CustomException
from src.utils import read_yaml

# Ensure resources are available
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)


def preprocess_comment(comment: str, params) -> str:
    """Apply preprocessing transformations to a comment string."""
    try:
        if pd.isna(comment) or not isinstance(comment, str):
            return ""

        if params.data_preprocessing.lowercase:
            comment = comment.lower().strip()

        comment = re.sub(r"\n", " ", comment)

        if params.data_preprocessing.remove_punct:
            comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

        stop_words = set(stopwords.words("english")) - set(params.data_preprocessing.stopwords_keep)
        words = [w for w in comment.split() if w not in stop_words]

        if params.data_preprocessing.lemmatize:
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(w) for w in words]

        return " ".join(words)

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise CustomException(e, sys)


def normalize_text(df: pd.DataFrame, params) -> pd.DataFrame:
    """Normalize the text column in the dataframe."""
    try:
        if "clean_comment" not in df.columns:
            raise ValueError("Input dataframe must contain 'clean_comment' column")

        original_shape = df.shape
        df = df.copy()
        df["clean_comment"] = df["clean_comment"].astype(str).apply(lambda x: preprocess_comment(x, params))

        logger.info(f"Text normalization completed: Shape {original_shape} -> {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise CustomException(e, sys)


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save processed train and test datasets."""
    try:
        processed_data_path = Path(data_path)
        os.makedirs(processed_data_path, exist_ok=True)

        train_file = os.path.join(processed_data_path, "train_processed.csv")
        test_file = os.path.join(processed_data_path, "test_processed.csv")

        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)

        logger.info(f"Processed train data saved: {train_file} | Shape: {train_data.shape}")
        logger.info(f"Processed test data saved: {test_file} | Shape: {test_data.shape}")

    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise CustomException(e, sys)


def initiate_data_preprocessing() -> None:
    """Pipeline entrypoint for preprocessing stage."""
    try:
        params = read_yaml(Path("params.yaml"))

        train_path = os.path.join(params.data_ingestion.data_path, "train.csv")
        test_path = os.path.join(params.data_ingestion.data_path, "test.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logger.info(f"Loaded train data: {train_path} | Shape: {train_df.shape}")
        logger.info(f"Loaded test data: {test_path} | Shape: {test_df.shape}")

        train_processed = normalize_text(train_df, params)
        test_processed = normalize_text(test_df, params)

        save_data(train_processed, test_processed, params.data_preprocessing.data_path)

        logger.info("Data preprocessing pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Error in data preprocessing pipeline: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    initiate_data_preprocessing()
