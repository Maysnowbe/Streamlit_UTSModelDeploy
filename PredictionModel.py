import streamlit as st
import pickle
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

def main():

    st.title("Churn Prediction")
    st.write("This Apps is used to predict customer churn probabilty baseed on their profile data.")

    credit_score = st.number_input("Credit Score:")
    geography = st.selectbox("Geography:", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender:", ["Male", "Female"])
    age = st.number_input("Age:")
    tenure = st.number_input("Tenure (years with company):")
    balance = st.number_input("Balance (account balance):")
    num_of_products = st.number_input("Number of Products:")
    has_cr_card = st.selectbox("Has Credit Card? (Yes/No)", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member? (Yes/No)", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary:")

    userData = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [1 if has_cr_card == "Yes" else 0],
        "IsActiveMember": [1 if is_active_member == "Yes" else 0],
        "EstimatedSalary": [estimated_salary]
    })
    print(userData)

    def loadEncoder(encoder_path="OneHotEncoder.pkl"):
        with open(encoder_path, "rb") as encoder_file:
            encoder = pickle.load(encoder_file)
        return encoder

    if st.button("Predict Churn Risk"):
        all_filled = credit_score and geography and gender and age and tenure and balance and num_of_products and has_cr_card and is_active_member and estimated_salary
        if not all_filled:
            st.error("Please fill all fields before submitting.")
            return
        
        encoder = loadEncoder()

        categorical = ['Geography', 'Gender']
        conti = ['CreditScore', 'Balance', 'EstimatedSalary']
        
        user_data_subset = userData[categorical]
        user_data_encoded = pd.DataFrame(encoder.transform(user_data_subset).toarray(), columns=encoder.get_feature_names_out(categorical))
        userData = userData.reset_index(drop=True)
        userData = pd.concat([userData, user_data_encoded], axis=1)
        userData.drop(categorical, axis=1, inplace=True)

        with st.spinner("Predicting... Please Wait"):
            # Load the pickled model from its saved location
            with open("finalized_model.pkl", "rb") as model_file:
                model = pickle.load(model_file)

            # Make prediction
            prediction = model.predict(userData)[0]  # Assuming prediction is a probability

            # Display prediction with clear interpretation
            if prediction == 1:
                st.write("Predicted: **CHURN**")
            else:
                st.write("Predicted: **NOT CHURN**")

if __name__ == "__main__":
    main()
