import streamlit as  st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

ps = PorterStemmer()
stop_words=set(stopwords.words('english'))

def transform_text(text):
    text=text.lower()
    tokens=nltk.word_tokenize(text)
    tokens = [w for w in tokens if w.isalnum()]
    tokens=[w for w in tokens if not w in stop_words]
    tokens=[ps.stem(word) for word in tokens]
    return " ".join(tokens)
tfidf=pickle.load(open("vectorizer.pkl", "rb"))
model=pickle.load(open("model.pkl", "rb"))


st.title("SMS Spam Detector")
st.write("Paste a Message to check whether it is a spam.")
msg= st.text_area("Enter the Message")

if st.button("Predict"):
    if msg.strip()==" ":
        st.warning("Please enter a message")
    else:
        vector=tfidf.transform([transform_text(msg)])
        result=model.predict(vector)[0]
        if result==1:
            st.error("Spam Detected")
        else :
            st.success("Not Spam (Ham)")
