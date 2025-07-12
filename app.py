import streamlit as st
import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# 🛠️ Page Config
st.set_page_config(page_title="📊 Churn Predictor", layout="centered")

# 🎨 Background Style
def set_background():
    page_bg_style = '''
    <style>
    body {
        background: linear-gradient(to right, #74ebd5, #ACB6E5);
        color: #0f2027;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: rgba(255,255,255,0.95);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 0 30px rgba(0,0,0,0.2);
    }
    </style>
    '''
    st.markdown(page_bg_style, unsafe_allow_html=True)

set_background()

# 🔍 Title
st.title("🔍 Customer Churn Prediction")
st.subheader("📈 Smart Dashboard for Data Analysts")

# 📦 Load Model and Tools
try:
    model = tf.keras.models.load_model('model_new.keras', compile=False)
    with open('label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        onehot_encoder_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    model_loaded = True
    st.success("✅ Model and encoders loaded successfully!")
except Exception as e:
    st.error(f"❌ Model or encoder loading failed:\n\n`{e}`")
    model_loaded = False

# 📋 Input Form
if model_loaded:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('🧑 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 30)
    balance = st.number_input('💰 Balance', value=0.0)
    credit_score = st.number_input('💳 Credit Score', value=650)
    estimated_salary = st.number_input('📊 Estimated Salary', value=50000.0)
    tenure = st.slider('📅 Tenure (Years)', 0, 10, 3)
    num_of_products = st.slider('🛍️ Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
    is_active_member = st.selectbox('🔥 Is Active Member?', [0, 1])

    # 🧠 Process Input
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    input_data_scaled = scaler.transform(input_data)

    # 🎯 Make Prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.markdown("### 🎯 Prediction Result")
    if prediction_proba > 0.5:
        st.error(f'⚠️ The customer is likely to churn.\n\n**Churn Probability:** `{prediction_proba:.2f}`')
    else:
        st.success(f'✅ The customer is not likely to churn.\n\n**Churn Probability:** `{prediction_proba:.2f}`')
else:
    st.info("🛑 Please ensure all files (`model_new.keras`, `.pkl`) are present in the same folder.")
