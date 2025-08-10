import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import os

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .stTitle {
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .high-risk {
        background-color: #ffcccc;
    }
    .low-risk {
        background-color: #ccffcc;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title('🏦 Customer Churn Prediction')
st.markdown("""
This app predicts the probability of a bank customer churning (leaving the bank) based on various factors.
Enter the customer details below to get a prediction.
""")

# Helper function to load models
@st.cache_resource
def load_models():
    # Load the trained model
    model = tf.keras.models.load_model('model.h5')
    
    # Load the encoders and scaler
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
        
    return model, label_encoder_gender, onehot_encoder_geo, scaler

# Load models
model, label_encoder_gender, onehot_encoder_geo, scaler = load_models()

# Create two columns for the form
col1, col2 = st.columns(2)

# User input form
with st.form("prediction_form"):
    st.subheader("Customer Information")
    
    # Column 1
    with col1:
        geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
        age = st.slider('Age', 18, 92, 35)
        tenure = st.slider('Tenure (Years)', 0, 10, 5)
        balance = st.number_input('Balance', min_value=0.0, value=50000.0, step=1000.0)
    
    # Column 2
    with col2:
        credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
        estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0, step=1000.0)
        num_of_products = st.slider('Number of Products', 1, 4, 1)
        has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    # Submit button
    submit_button = st.form_submit_button(label="Predict Churn")

# Make prediction when form is submitted
if submit_button:
    # Show spinner during calculation
    with st.spinner("Calculating prediction..."):
        # Prepare the input data
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

        # One-hot encode 'Geography'
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

        # Combine one-hot encoded columns with input data
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Predict churn
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]
        
        # Display prediction
        st.subheader("Prediction Result")
        
        # Create a progress bar for the probability
        st.progress(float(prediction_proba))
        
        # Display the probability as a percentage
        risk_class = "high-risk" if prediction_proba > 0.5 else "low-risk"
        
        # Show prediction box with appropriate color
        st.markdown(f"""
        <div class="prediction-box {risk_class}">
            <h3>Churn Probability: {prediction_proba:.2%}</h3>
            <p>{'The customer is likely to churn.' if prediction_proba > 0.5 else 'The customer is not likely to churn.'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional information based on prediction
        if prediction_proba > 0.5:
            st.warning("⚠️ This customer is at high risk of churning! Consider reaching out with retention offers.")
            
            # Show factors that might be influencing the prediction
            st.subheader("Possible risk factors:")
            factors = []
            if age > 60:
                factors.append("- Customer age is high")
            if balance < 10000:
                factors.append("- Low account balance")
            if is_active_member == 0:
                factors.append("- Customer is not an active member")
            if tenure < 3:
                factors.append("- Short tenure with bank")
            
            if factors:
                for factor in factors:
                    st.markdown(factor)
            else:
                st.markdown("Multiple factors contribute to this prediction.")
        else:
            st.success("✅ This customer is likely to stay! They appear satisfied with your services.")

# Add information about the model
with st.expander("About this model"):
    st.markdown("""
    This prediction model is built using a Neural Network with the following characteristics:
    - **Architecture**: Multi-layer perceptron with ReLU activation functions
    - **Input features**: Customer demographics and account information
    - **Output**: Probability of customer churning
    
    The model was trained on historical bank customer data with features like credit score,
    geography, gender, age, tenure, balance, and more.
    """)
