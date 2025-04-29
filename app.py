import streamlit as st
import joblib
import re

# Load the saved TF-IDF vectorizer and trained model
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("fakenews_model.pkl")

# Simple text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.strip()

# Set up Streamlit UI
st.title("📰 Fake News Detector")
st.subheader("Paste a news article or headline below and find out if it's REAL or FAKE.")

# User input
user_input = st.text_area("Enter News Text:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Clean the input
        cleaned_input = clean_text(user_input)

        # Preprocess and transform input
        input_vectorized = tfidf.transform([cleaned_input])

        # Predict
        prediction = model.predict(input_vectorized)[0]
        prediction_proba = model.predict_proba(input_vectorized)[0]  # [prob_fake, prob_real]

        # Display main prediction
        if prediction == 0:
            st.error(f"🛑 Prediction: **FAKE NEWS** ({prediction_proba[0] * 100:.2f}% confidence)")
        else:
            st.success(f"✅ Prediction: **REAL NEWS** ({prediction_proba[1] * 100:.2f}% confidence)")

        # Display both probabilities nicely
        st.markdown("### 🔍 Prediction Confidence Breakdown:")

        # Custom HTML for prettier horizontal bars
        fake_news_percentage = prediction_proba[0] * 100
        real_news_percentage = prediction_proba[1] * 100

        # Fake News Bar (Red)
        st.markdown(
            f"""
            <div style="background-color:#FFCCCC; padding:5px; border-radius:5px;">
                <b>🛑 Fake News:</b> {fake_news_percentage:.2f}% 
                <div style="background-color:#FF0000; width:{fake_news_percentage}%; height:20px; border-radius:5px;"></div>
            </div>
            """, unsafe_allow_html=True
        )

        # Real News Bar (Green)
        st.markdown(
            f"""
            <div style="background-color:#CCFFCC; padding:5px; border-radius:5px; margin-top:10px;">
                <b>✅ Real News:</b> {real_news_percentage:.2f}% 
                <div style="background-color:#00CC00; width:{real_news_percentage}%; height:20px; border-radius:5px;"></div>
            </div>
            """, unsafe_allow_html=True
        )
