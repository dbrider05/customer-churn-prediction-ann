#Stream lit Web App
import streamlit as st 
import pandas as pd 
import numpy as np  
import tensorflow as tf
import pickle 
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder

#Load the trained model
model = tf.keras.models.load_model('model.h5')

#Load the encoders and scalers

with open('geo_one_hot_encoder.pkl','rb') as file:
    geo_one_hot_encoder = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

## streamlit app

st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn or not based on their information.")
st.write("Please enter the following information:")

#Input fields

geography = st.selectbox("Geography", geo_one_hot_encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age  = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


#prepare the input data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary],
})

#One hot encode the geography column
geo_encoded = geo_one_hot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_one_hot_encoder.get_feature_names_out(['Geography']))

# Concatenate the encoded geography columns with the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction

prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]


if st.button("Predict Churn"):

    st.write(f"Churn Probability: {prediction_prob:.2f}")

    if prediction_prob > 0.5:
        st.warning("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is not likely to churn.")
