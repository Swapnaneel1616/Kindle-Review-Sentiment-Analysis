# Kindle Review Sentiment Analysis

This project performs sentiment analysis on Amazon Kindle customer reviews using Natural Language Processing (NLP) and Machine Learning techniques. The goal is to automatically classify reviews into sentiment categories (positive or negative) and extract meaningful insights from large volumes of textual data.

## 📌 Project Overview

Customer reviews contain valuable feedback but are unstructured and difficult to analyze at scale. This project builds an end-to-end NLP pipeline that cleans review text, converts it into numerical features using TF-IDF, and trains a machine learning model to predict sentiment.

## 🛠️ Technologies & Libraries Used

The following Python libraries and packages are used in this project:

* **NumPy** – numerical operations
* **Pandas** – data manipulation and analysis
* **Matplotlib** & **Seaborn** – data visualization
* **BeautifulSoup (bs4)** – HTML/text cleaning
* **NLTK** – text preprocessing (tokenization, stopwords, lemmatization)
* **Scikit-learn** – feature extraction, model building, evaluation

### Key Modules Used

* `TfidfVectorizer`
* `Train-test split`
* `Gaussian Naive Bayes`
* `Pipeline`
* `StandardScaler`
* `Accuracy Score, Confusion Matrix, Classification Report`

## 🔄 Workflow

1. **Data Loading & Cleaning**

   * Remove HTML tags and special characters
   * Convert text to lowercase

2. **Text Preprocessing**

   * Tokenization (words & sentences)
   * Stopword removal
   * Lemmatization

3. **Feature Engineering**

   * Convert text into numerical form using TF-IDF Vectorization

4. **Model Training**

   * Split data into training and testing sets
   * Train a Gaussian Naive Bayes classifier

5. **Evaluation**

   * Measure performance using accuracy, confusion matrix, and classification report

## 📊 Results

The trained model successfully classifies customer reviews based on sentiment, demonstrating the effectiveness of NLP techniques combined with traditional machine learning algorithms.

## 🚀 Future Improvements

* Add deep learning models (LSTM, BERT)
* Perform multi-class sentiment analysis
* Deploy as a web application using Flask or FastAPI

## 📂 Domain

Natural Language Processing | Machine Learning | Text Analytics

---

This project showcases practical NLP skills and is suitable for academic learning, portfolio projects, and real-world sentiment analysis use cases.
