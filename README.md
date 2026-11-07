# SMS Spam Detector

Classifies SMS messages as spam or ham using classical machine learning.

- **Dataset:** SMS Spam Collection (UCI), 5,572 labeled messages
- **Preprocessing:** lowercase, tokenize, remove stopwords, Porter stemming
- **Features:** TF-IDF (3,000 features)
- **Models compared:** Naive Bayes, Logistic Regression, SVM
- **Best model:** SVM (accuracy 99.2%, precision 1.00, recall 0.94, F1 0.97)
- **App:** Streamlit

## Run locally

    pip install -r requirements.txt
    streamlit run app.py
