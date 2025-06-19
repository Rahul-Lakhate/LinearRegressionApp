import streamlit as st
import numpy as np
from joblib import load

# Load the trained model
model = load("titanic_model.joblib")

# Streamlit UI
st.set_page_config(page_title="Titanic Survival Predictor")
st.title("🚢 Titanic Survival Prediction App")

st.markdown("""
This app predicts whether a passenger would survive the Titanic disaster based on input features.
""")

# Input fields
Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 100, 25)
SibSp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
Parch = st.number_input("Number of Parents/Children Aboard (Parch)", 0, 10, 0)
Fare = st.number_input("Ticket Fare", 0.0, 600.0, 50.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode categorical inputs
Sex_encoded = 1 if Sex == "male" else 0
Embarked_mapping = {"C": 0, "Q": 1, "S": 2}
Embarked_encoded = Embarked_mapping[Embarked]

# Combine inputs
features = np.array([[Pclass, Sex_encoded, Age, SibSp, Parch, Fare, Embarked_encoded]])

# Predict button
if st.button("Predict Survival"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.success(f"✅ Survived (Probability: {probability:.2%})")
    else:
        st.error(f"❌ Did not survive (Probability: {probability:.2%})")
