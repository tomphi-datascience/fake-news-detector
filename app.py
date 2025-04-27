import streamlit as st
import joblib

# Load the saved TF-IDF vectorizer and Random Forest model
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("fakenews_model.pkl")

# Set up Streamlit UI
st.title("📰 Fake News Detector")
st.subheader("Paste a news article or headline below and find out if it's REAL or FAKE.")

# User input
user_input = st.text_area("Enter News Text:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Preprocess and transform input
        input_vectorized = tfidf.transform([user_input])

        # Predict
        prediction = model.predict(input_vectorized)[0]
        prediction_proba = model.predict_proba(input_vectorized)

        # Display result
        if prediction == 0:
            st.error(f"🛑 Prediction: **FAKE NEWS**")
        else:
            st.success(f"✅ Prediction: **REAL NEWS**")
        
        # Optional: Show confidence
        st.write(f"Confidence (Fake vs Real): {prediction_proba[0][0]:.2f} / {prediction_proba[0][1]:.2f}")
