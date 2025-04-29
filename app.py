import streamlit as st
import joblib

# Load model and vectorizer
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
        # Minimal cleaning: just strip spaces
        cleaned_input = user_input.strip()

        # Vectorize input
        input_vectorized = tfidf.transform([cleaned_input])

        # Predict
        prediction = model.predict(input_vectorized)[0]
        prediction_proba = model.predict_proba(input_vectorized)[0]

        # Show main prediction
        confidence = prediction_proba[prediction] * 100  # get confidence of predicted class
        if prediction == 0:
            st.error(f"🛑 Prediction: **FAKE NEWS** ({confidence:.2f}% confidence)")
        else:
            st.success(f"✅ Prediction: **REAL NEWS** ({confidence:.2f}% confidence)")

        # Show both probabilities simply
        st.markdown("### 🔍 Confidence Scores:")
        st.write(f"🛑 **Fake News Probability:** {prediction_proba[0] * 100:.2f}%")
        st.write(f"✅ **Real News Probability:** {prediction_proba[1] * 100:.2f}%")
