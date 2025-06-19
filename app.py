import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("titanic_model.pkl", "rb"))

st.title("🚢 Titanic Survival Predictor")

# Input fields
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 100, 25)
SibSp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
Fare = st.number_input("Fare", 0.0, 600.0, 50.0)
Embarked = st.selectbox("Embark Port", ["C", "Q", "S"])

# Encode
Sex = 1 if Sex == "male" else 0
Embarked = {"C": 0, "Q": 1, "S": 2}[Embarked]

# Predict
features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])

if st.button("Predict"):
    result = model.predict(features)
    st.success("✅ Survived!" if result[0] == 1 else "❌ Did not survive.")
