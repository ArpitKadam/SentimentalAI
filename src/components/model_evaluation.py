import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src.logger import logger
from src.exception import CustomException
from src.utils import read_yaml, load_data_for_model, load_model, load_vectorizer, create_directories
import dagshub
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI')
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

dagshub.init(repo_owner='ArpitKadam', repo_name='SentimentalAI', mlflow=True)

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and return metrics."""
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr', average='weighted')
        logger.debug("Model evaluation completed")
        return report, cm, acc, f1, precision, recall, roc_auc
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise CustomException(e, sys)

def log_confusion_matrix(cm, save_path: str):
    """Log confusion matrix as artifact."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(save_path)
        plt.close()
        logger.debug(f"Confusion matrix saved: {save_path}")
    except Exception as e:
        logger.error(f"Error saving confusion matrix: {e}")
        raise CustomException(e, sys)

def save_json(data: dict, file_path: str):
    """Save dictionary to JSON."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.debug(f"JSON saved: {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")
        raise CustomException(e, sys)

def main():
    try:
        # Load params
        params_all = read_yaml(Path("params.yaml"))
        params_model = params_all.model_building
        params_eval = params_all.model_evaluation
        eval_dir = params_eval.data_path
        create_directories([eval_dir], verbose=True)

        # Load model and vectorizer
        model_file = os.path.join(params_model.model_path, "lgbm_model.pkl")
        vectorizer_file = os.path.join(params_model.model_path, "tfidf_vectorizer.pkl")
        model = load_model(model_file)
        vectorizer = load_vectorizer(vectorizer_file)

        # Load test data
        test_file = os.path.join("Artifacts/data_preprocessing", "test_processed.csv")
        test_data = load_data_for_model(test_file)
        X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
        y_test = test_data['category'].values

        # Start MLflow run
        mlflow.set_experiment("dvc-pipeline-runs")
        with mlflow.start_run() as run:
            # Log model parameters
            for key, value in params_model.items():
                mlflow.log_param(key, value)

            # Signature inference
            input_example = pd.DataFrame(X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out())
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))

            # Log model
            mlflow.sklearn.log_model(model, "lgbm_model", signature=signature, input_example=input_example)
            mlflow.log_artifact(vectorizer_file)

            # Evaluate model
            report, cm, acc, f1, precision, recall, roc_auc = evaluate_model(model, X_test_tfidf, y_test)

            # Save all metrics to JSON
            metrics_dict = {
                "accuracy": acc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc,
            }

            metrics_file = os.path.join(eval_dir, "metrics.json")
            save_json(metrics_dict, metrics_file)
            mlflow.log_artifact(metrics_file)

            report_file = os.path.join(eval_dir, "classification_report.json")
            save_json(report, report_file)
            mlflow.log_artifact(report_file)

            for key, value in metrics_dict.items():
                mlflow.log_metric(key, value)
            logger.info(f"Logged metrics: {metrics_dict}")

            # Save confusion matrix
            cm_file = os.path.join(eval_dir, "confusion_matrix.png")
            log_confusion_matrix(cm, cm_file)
            mlflow.log_artifact(cm_file)

            # Save model info JSON
            model_info_file = os.path.join(eval_dir, "model_info.json")
            save_json({"run_id": run.info.run_id, "model_path": run.info.artifact_uri}, model_info_file)
            mlflow.log_artifact(model_info_file)

            # Add tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

            logger.info("Model evaluation and artifact logging completed successfully.")

    except Exception as e:
        logger.error(f"Failed to complete model evaluation: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
