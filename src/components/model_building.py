import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from src.utils import read_yaml, load_data_for_model, save_model
from src.logger import logger
from src.exception import CustomException
import sys
from pathlib import Path

def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple, save_path: str):
    """Apply TF-IDF to the training data and save the vectorizer."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        logger.info(f"TF-IDF transformation complete. Train shape: {X_train_tfidf.shape}")

        # Save the vectorizer
        os.makedirs(save_path, exist_ok=True)
        vectorizer_file = os.path.join(save_path, "tfidf_vectorizer.pkl")
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.info(f"TF-IDF vectorizer saved to {vectorizer_file}")
        return X_train_tfidf, y_train

    except Exception as e:
        logger.error(f"Error during TF-IDF transformation: {e}")
        raise CustomException(e, sys)


def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, params: dict):
    """Train a LightGBM model."""
    try:
        model = lgb.LGBMClassifier(
            objective=params.get("objective", "multiclass"),
            num_class=params.get("num_class", 3),
            metric=params.get("metric", "multi_logloss"),
            is_unbalance=params.get("is_unbalance", True),
            class_weight=params.get("class_weight", "balanced"),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 0.1),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 10),
            n_estimators=params.get("n_estimators", 100),
            random_state=params.get("random_state", 42),
            num_leaves=params.get("num_leaves", 31)
        )
        model.fit(X_train, y_train)
        logger.info("LightGBM model training completed")
        return model
    except Exception as e:
        logger.error(f"Error during LightGBM model training: {e}")
        raise CustomException(e, sys)


def main():
    try:
        # Load parameters from params.yaml
        params = read_yaml(Path("params.yaml")).model_building

        model_path = params.get("model_path", "Artifacts/model_building")
        os.makedirs(model_path, exist_ok=True)

        # Load preprocessed training data
        train_data_file = os.path.join("Artifacts/data_preprocessing", "train_processed.csv")
        train_data = load_data_for_model(train_data_file)

        # Apply TF-IDF
        X_train_tfidf, y_train = apply_tfidf(
            train_data,
            max_features=params["max_features"],
            ngram_range=tuple(params["ngram_range"]),
            save_path=model_path
        )

        # Train LightGBM
        lgbm_model = train_lgbm(X_train_tfidf, y_train, params)

        # Save trained model
        save_model(lgbm_model, os.path.join(model_path, "lgbm_model.pkl"))

        logger.info("Model building pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Failed to complete model building: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
