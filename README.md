# SMS Spam Detection Using Classical Machine Learning

A spam classifier for SMS messages built with TF-IDF features and classical ML models, deployed as a Streamlit web app.

**Live demo:** _add your Streamlit link here after deployment_

## Overview

The project compares three classical models on the SMS Spam Collection dataset and deploys the best one. Because spam is much rarer than ham, models are judged on precision, recall, and F1, not accuracy alone. In a spam filter, blocking a real message is the worst error, so precision is prioritised.

## Dataset

- **Source:** SMS Spam Collection (UCI Machine Learning Repository)
- **Size:** 5,572 labeled messages (`ham` / `spam`)
- **Duplicates:** removed before training
- **Imbalance:** about 12.6% of messages are spam

## Pipeline

1. Load data and remove duplicates
2. Encode labels (ham = 0, spam = 1) and explore message lengths
3. Preprocess text: lowercase, tokenize, remove punctuation and stopwords, Porter stemming
4. Split 80/20 with stratification (`random_state=2`)
5. TF-IDF vectorization (3,000 features), fit on training data only
6. Train and compare Naive Bayes, Logistic Regression, and SVM
7. Evaluate with accuracy, precision, recall, F1, ROC-AUC, confusion matrices, TPR/FPR
8. Save the best model and vectorizer with pickle
9. Serve predictions through Streamlit

## Results

Test set: 1,034 messages (903 ham, 131 spam).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Naive Bayes | 0.9787 | 1.0000 | 0.8321 | 0.9083 | 0.9854 |
| Logistic Regression | 0.9710 | 0.9903 | 0.7786 | 0.8718 | 0.9842 |
| **SVM (LinearSVC)** | **0.9923** | **1.0000** | **0.9389** | **0.9685** | 0.9842 |

Confusion matrix counts:

| Model | True Neg | False Pos | False Neg | True Pos |
|---|---|---|---|---|
| Naive Bayes | 903 | 0 | 22 | 109 |
| Logistic Regression | 902 | 1 | 29 | 102 |
| SVM | 903 | 0 | 8 | 123 |

**Best model: SVM.** It blocked no real messages (false positive rate 0) and caught 123 of 131 spam messages (true positive rate 0.939), with the highest F1.

## Project structure

```
app.py              Streamlit app
model.pkl           Trained SVM model
vectorizer.pkl      Fitted TF-IDF vectorizer
requirements.txt    Pinned dependencies
README.md           Project documentation
```

## Run locally

```bash
git clone https://github.com/Suleiman505/-Sms-Spam-Classifier.git
cd -Sms-Spam-Classifier
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Tech stack

Python, pandas, scikit-learn, NLTK, Streamlit

## Limitations

- Trained on an English-language dataset from the 2000s, so modern spam styles (for example links and crypto scams) may be missed
- TF-IDF ignores word order and meaning
- The dataset is small and imbalanced

## Future work

- Compare TF-IDF settings (n-grams, vocabulary size)
- Analyse false negatives to find missed spam patterns
- Try class weighting or threshold tuning to improve recall
- Test on newer, more diverse SMS data

## Author

Suleiman, [GitHub: Suleiman505](https://github.com/Suleiman505)