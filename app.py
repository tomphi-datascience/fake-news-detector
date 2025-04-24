import streamlit as st
from sklearn.externals import joblib

model = joblib.load("fakenews_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article or headline below to check if it's **FAKE** or **REAL**.")

user_input = st.text_area("Paste your news content here:", height=200)

if st.button("Predict"):
    if user_input.strip():
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        confidence = model.predict_proba(input_vector)[0][prediction]

        label = "🟩 REAL" if prediction == 1 else "🟥 FAKE"
        st.markdown(f"### Prediction: {label}")
        st.write(f"**Confidence:** {round(confidence * 100, 2)}%")
    else:
        st.warning("Please enter some text before clicking Predict.")
