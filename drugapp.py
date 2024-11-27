import streamlit as st
import joblib
import pandas as pd

# Load pre-trained model, TF-IDF vectorizer, and drug recommendations
@st.cache_resource
def load_saved_artifacts():
    gb = joblib.load("gb.pkl")
    tf = joblib.load("tf.pkl")
    drug_recommendations = joblib.load("drug_recommendations.pkl")
    return gb, tf, drug_recommendations

gb, tf, drug_recommendations = load_saved_artifacts()

# Streamlit App
st.title("Drug Recommendation System")
st.write("Enter a review to predict the condition and get a drug recommendation.")

# Text input for user review
user_review = st.text_area("Enter the drug review:", "")

if st.button("Predict Condition and Recommend Drug"):
    if user_review:
        # Transform review and predict condition
        review_vector = tf.transform([user_review])  # No preprocessing
        predicted_condition = gb.predict(review_vector)[0]

        # Recommend drug
        recommended_drug = drug_recommendations.get(predicted_condition, "No recommendation available")

        # Display results
        st.write(f"**Predicted Condition:** {predicted_condition}")
        st.write(f"**Recommended Drug:** {recommended_drug}")
    else:
        st.write("Please enter a review to proceed.")
