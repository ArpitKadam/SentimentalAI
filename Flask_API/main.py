import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.dates as mdates
from flask_cors import CORS
import re
import pickle
import mlflow
import dagshub
import os
import sys
import traceback
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from src.logger import logger
from src.exception import CustomException

# Load environment variables
load_dotenv()

try:
    os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI')
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

    dagshub.init(repo_owner='ArpitKadam', repo_name='SentimentalAI', mlflow=True)
    logger.info("Dagshub and MLflow tracking initialized successfully.")
except Exception as e:
    logger.error("Failed to initialize Dagshub/MLflow.", exc_info=True)
    raise CustomException(e, sys)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

def preprocess_comment(comment: str) -> str:
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)  # remove newlines
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)  # keep alphanum + punctuation

        # Remove stopwords but keep negation-related ones
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        tokens = [word for word in comment.split() if word not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return ' '.join(tokens)

    except Exception as e:
        logger.error("Error during comment preprocessing.", exc_info=True)
        raise CustomException(e, sys)


def load_model_and_vectorizer(model_version: str, model_name: str, vectorizer_path: str):
    """Load MLflow model and TF-IDF vectorizer."""
    try:
        model_uri = f"runs:/{model_version}/{model_name}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model loaded successfully from {model_uri}")

        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")

        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.info("TF-IDF vectorizer loaded successfully.")

        return model, vectorizer

    except Exception as e:
        logger.error("Failed to load model or vectorizer.", exc_info=True)
        raise CustomException(e, sys)


# Load model/vectorizer once at startup
try:
    model, vectorizer = load_model_and_vectorizer(
        "9c619d2a8e284470a10dee9818c906a8",
        "lgbm_model",
        "Artifacts/model_building/tfidf_vectorizer.pkl"
    )
except Exception as e:
    logger.error("Critical: Model/vectorizer could not be loaded.", exc_info=True)
    model, vectorizer = None, None


@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Sentimental AI API!"})


@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment from comments text using MLflow schema alignment."""
    try:
        # Ensure model and vectorizer are loaded
        if model is None or vectorizer is None:
            logger.error("Model or vectorizer not loaded.")
            return jsonify({"error": "Model or vectorizer not loaded."}), 500

        # Validate JSON input
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request must be JSON with a 'comments' field."}), 400

        comments = data.get("comments")
        if not isinstance(comments, list) or not comments:
            return jsonify({"error": "Field 'comments' must be a non-empty list."}), 400

        # Preprocess
        preprocessed_comments = [preprocess_comment(c) for c in comments]

        # Vectorize
        transformed = vectorizer.transform(preprocessed_comments)
        vectorizer_features = vectorizer.get_feature_names_out()

        # Convert to DataFrame with vectorizer features
        df_input = pd.DataFrame(transformed.toarray(), columns=vectorizer_features)

        # Align with MLflow model schema
        expected_features = list(model.metadata.get_input_schema().input_names())
        df_input = df_input.reindex(columns=expected_features, fill_value=0)

        # Predict
        predictions = model.predict(df_input).tolist()
        
        # Response
        response = [
            {"comment": c, "prediction": p}
            for c, p in zip(comments, predictions)
        ]
        return jsonify(response), 200

    except Exception as e:
        logger.error("Prediction failed.", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    """Predict sentiment from comments with timestamps."""
    try:
        if model is None or vectorizer is None:
            return jsonify({"error": "Model or vectorizer not loaded."}), 500

        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request must be JSON with 'comments' field."}), 400

        comments_data = data.get('comments')
        if not isinstance(comments_data, list) or not comments_data:
            return jsonify({"error": "Field 'comments' must be a non-empty list of {text, timestamp}."}), 400

        comments = [item.get('text') for item in comments_data if 'text' in item]
        timestamps = [item.get('timestamp') for item in comments_data if 'timestamp' in item]

        if not comments or not timestamps:
            return jsonify({"error": "Each comment must include 'text' and 'timestamp'."}), 400

        # Preprocess
        preprocessed_comments = [preprocess_comment(c) for c in comments]

        # Vectorize & schema align
        transformed = vectorizer.transform(preprocessed_comments)
        vectorizer_features = vectorizer.get_feature_names_out()
        df_input = pd.DataFrame(transformed.toarray(), columns=vectorizer_features)
        expected_features = list(model.metadata.get_input_schema().input_names())
        df_input = df_input.reindex(columns=expected_features, fill_value=0)

        predictions = model.predict(df_input).tolist()

        response = [
            {"comment": c, "sentiment": p, "timestamp": t}
            for c, p, t in zip(comments, predictions, timestamps)
        ]
        return jsonify(response), 200

    except Exception as e:
        logger.error("Prediction with timestamps failed.", exc_info=True)
        return jsonify({"error": f"Prediction with timestamps failed: {str(e)}"}), 500


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    """Generate a sentiment distribution chart."""
    try:
        data = request.get_json(silent=True)
        sentiment_counts = data.get('sentiment_counts')

        if not sentiment_counts or not isinstance(sentiment_counts, dict):
            return jsonify({"error": "No valid sentiment counts provided."}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            return jsonify({"error": "Sentiment counts sum to zero."}), 400

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        logger.error("Chart generation failed.", exc_info=True)
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    """Generate a word cloud from comments."""
    try:
        data = request.get_json(silent=True)
        comments = data.get('comments')

        if not isinstance(comments, list) or not comments:
            return jsonify({"error": "Field 'comments' must be a non-empty list."}), 400

        preprocessed_comments = [preprocess_comment(c) for c in comments]
        text = ' '.join(preprocessed_comments)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')) - {'not', 'no', 'but', 'however', 'yet'},
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        logger.error("Word cloud generation failed.", exc_info=True)
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    """Generate a sentiment trend graph over time."""
    try:
        data = request.get_json(silent=True)
        sentiment_data = data.get('sentiment_data')

        if not isinstance(sentiment_data, list) or not sentiment_data:
            return jsonify({"error": "Field 'sentiment_data' must be a non-empty list."}), 400

        df = pd.DataFrame(sentiment_data)
        if not {'timestamp', 'sentiment'}.issubset(df.columns):
            return jsonify({"error": "Each record must include 'timestamp' and 'sentiment'."}), 400

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        monthly_counts = df.resample('ME')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: 'red', 0: 'gray', 1: 'green'}

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        logger.error("Trend graph generation failed.", exc_info=True)
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    except Exception as e:
        logger.error("Flask app failed to start.", exc_info=True)
        raise CustomException(e, sys)