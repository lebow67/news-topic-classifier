import re
import joblib
import gradio as gr
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources for local/deployed environments
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

model = joblib.load("news_topic_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# ============================================================
# NLP PREPROCESSING
# ============================================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """
    Clean and preprocess input text using the same
    preprocessing steps used during model training.
    """

    # Remove punctuation, numbers and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = text.split()

    # Remove stopwords and perform lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    # Convert tokens back into a string
    return " ".join(tokens)


# ============================================================
# TOPIC INFORMATION
# ============================================================

topic_info = {
    1: {
        "name": "🌍 World",
        "description": "International news, politics, countries and global events."
    },
    2: {
        "name": "⚽ Sports",
        "description": "Sports events, teams, players, matches and competitions."
    },
    3: {
        "name": "💼 Business",
        "description": "Companies, markets, finance, economy and business activities."
    },
    4: {
        "name": "🔬 Science / Technology",
        "description": "Science, technology, computing, research and innovation."
    }
}


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_topic(text):

    # Check empty input
    if not text or not text.strip():
        return "⚠️ Please enter a news article or headline.", {}

    # Preprocess input
    processed_text = preprocess_text(text)

    # Check if preprocessing removed everything
    if not processed_text:
        return "⚠️ Please enter meaningful text.", {}

    # Convert text to TF-IDF features
    vectorized_text = tfidf.transform([processed_text])

    # Predict class
    prediction = model.predict(vectorized_text)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(vectorized_text)[0]

    # Create probability dictionary
    confidence = {}

    for class_id, probability in zip(model.classes_, probabilities):
        topic_name = topic_info[class_id]["name"]
        confidence[topic_name] = float(probability)

    # Get predicted topic
    predicted_topic = topic_info[prediction]["name"]

    result = (
        f"### {predicted_topic}\n\n"
        f"**Confidence: {probabilities[list(model.classes_).index(prediction)] * 100:.2f}%**\n\n"
        f"{topic_info[prediction]['description']}"
    )

    return result, confidence


# ============================================================
# CLEAR FUNCTION
# ============================================================

def clear_fields():
    return "", "Prediction will appear here.", {}


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(title="News Topic Classifier") as demo:

    gr.Markdown(
        """
        # 📰 News Topic Classifier

        ### AI-powered news classification using NLP and Machine Learning

        Enter a news headline or article and the model will classify it
        into one of four categories.
        """
    )

    gr.Markdown(
        """
        **Available Topics**

        🌍 **World** &nbsp;&nbsp;
        ⚽ **Sports** &nbsp;&nbsp;
        💼 **Business** &nbsp;&nbsp;
        🔬 **Science / Technology**
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            news_input = gr.Textbox(
                label="News Article / Headline",
                placeholder="Enter a news headline or article here...",
                lines=12
            )

            with gr.Row():
                predict_button = gr.Button("🔍 Classify News", variant="primary")
                clear_button = gr.Button("🗑️ Clear")

        with gr.Column(scale=2):
            prediction_output = gr.Markdown(value="Prediction will appear here.")
            confidence_output = gr.Label(
                label="Topic Confidence",
                num_top_classes=4
            )

    gr.Markdown("## 💡 Try an Example")

    gr.Examples(
        examples=[
            ["The United Nations announced a new international agreement between several countries to improve global cooperation."],
            ["The football team secured a dramatic victory in the championship final after scoring in the final minutes."],
            ["The company reported strong quarterly earnings as revenue increased significantly compared with last year."],
            ["Researchers have developed a new artificial intelligence system capable of analyzing complex scientific data."]
        ],
        inputs=news_input
    )

    gr.Markdown(
        """
        ## ⚙️ How It Works

        **1. Text Input** — The user enters a news headline or article.

        **2. NLP Preprocessing** — The text is cleaned, tokenized, stopwords are removed, and words are lemmatized.

        **3. TF-IDF Vectorization** — The processed text is converted into numerical features.

        **4. Logistic Regression** — The trained machine learning model predicts the news category.

        **5. Confidence Scores** — The application displays the probability associated with each topic.
        """
    )

    gr.Markdown(
        """
        ## 📊 Model Information

        | Component | Details |
        |---|---|
        | Dataset | AG News |
        | Task | Multi-class Text Classification |
        | Features | TF-IDF |
        | N-grams | Unigrams + Bigrams |
        | Features Used | 50,000 |
        | Classifier | Logistic Regression |
        | Test Accuracy | **92.05%** |
        | Classes | 4 |
        """
    )

    predict_button.click(
        fn=predict_topic,
        inputs=news_input,
        outputs=[prediction_output, confidence_output]
    )

    clear_button.click(
        fn=clear_fields,
        inputs=[],
        outputs=[news_input, prediction_output, confidence_output]
    )


if __name__ == "__main__":
    demo.launch()
