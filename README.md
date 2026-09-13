# 📰 News Topic Classifier

A machine learning NLP project that classifies news articles into four topics: **World, Sports, Business, and Science/Technology**.

## 🚀 Project Overview

This project uses the **AG News** dataset and a traditional NLP pipeline built with scikit-learn and NLTK.

### Model Pipeline

1. Combine news title and description
2. Convert text to lowercase
3. Remove punctuation and numbers
4. Normalize whitespace
5. Tokenize text
6. Remove English stopwords
7. Apply WordNet lemmatization
8. Convert text into TF-IDF features
9. Classify with Logistic Regression

## 📊 Dataset

The project uses the AG News dataset:

- Training samples: 120,000
- Test samples: 7,600
- Classes: 4
- Features: title + description

The original dataset files are intentionally not included in this repository.

## 🧠 Model

**TF-IDF Vectorization**

- Maximum features: 50,000
- N-gram range: unigram + bigram

**Classifier:** Logistic Regression

**Test Accuracy:** **92.05%**

### Classes

| Class | Topic |
|---|---|
| 1 | 🌍 World |
| 2 | ⚽ Sports |
| 3 | 💼 Business |
| 4 | 🔬 Science / Technology |

## 💻 Gradio App

The project includes a Gradio interface where you can enter a news headline or article and receive:

- Predicted topic
- Confidence scores for all four classes

## 📁 Project Structure

```text
news-topic-classifier/
├── app.py
├── news_topic_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── notebook/
    └── news_classifier.ipynb
```

## ⚙️ Run Locally

Clone the repository and install the dependencies:

```bash
git clone https://github.com/lebow67/news-topic-classifier.git
cd news-topic-classifier
pip install -r requirements.txt
python app.py
```

The Gradio application will provide a local URL in the terminal.

## 🛠️ Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib
- Gradio
- Jupyter Notebook

## 📌 Future Improvements

- Compare Logistic Regression with LinearSVC
- Add a validation split and hyperparameter tuning
- Improve Business vs Science/Technology classification
- Deploy the Gradio application
- Experiment with transformer-based NLP models

## 👤 Author

**Lebow67**
