import streamlit as st
import joblib

# Load the saved TF-IDF vectorizer and the trained model
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
        prediction_proba = model.predict_proba(input_vectorized)[0]  # [prob_fake, prob_real]

        # Display main prediction
        if prediction == 0:
            st.error(f"🛑 Prediction: **FAKE NEWS** ({prediction_proba[0] * 100:.2f}% confidence)")
        else:
            st.success(f"✅ Prediction: **REAL NEWS** ({prediction_proba[1] * 100:.2f}% confidence)")

        # Display both probabilities cleanly
        st.markdown("### 🔍 Prediction Confidence Breakdown:")

        # Display fake news probability
        st.write(f"**🛑 Fake News Probability:** {prediction_proba[0] * 100:.2f}%")
        st.progress(prediction_proba[0])  # between 0 and 1

        # Display real news probability
        st.write(f"**✅ Real News Probability:** {prediction_proba[1] * 100:.2f}%")
        st.progress(prediction_proba[1])  # between 0 and 1
