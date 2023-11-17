import pickle
import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model

model = load_model('Deployment.h5')
with open("my_scalar.pkl", "rb") as f:
    scaler = pickle.load(f)

def predict_churn(customer_inputs):
    st.title('Customer Churn Prediction')
    churn_probability = model.predict([customer_inputs])
    return churn_probability

def map_input_to_values(value, field):
    mappings = {
        'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
        'Gender': {'Female': 0, 'Male': 1},
        'PaymentMethod': {'Electronic check': 2, 'Mailed check': 3, 'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'TechSupport': {'No': 0, 'Yes': 2, 'No internet service': 1},
        'OnlineBackup': {'Yes': 2, 'No': 0, 'No internet service': 1}
    }
    return mappings.get(field, {}).get(value)

def get_user_input():
    feature_fields = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod', 'TechSupport', 'OnlineSecurity', 'Gender', 'OnlineBackup']
    user_input_values = {}
    
    for field in feature_fields:
        if field not in ['Contract', 'PaymentMethod', 'TechSupport', 'OnlineSecurity', 'Gender', 'OnlineBackup']:
            user_input_values[field] = st.number_input(field, value=0)
        else:
            user_input_values[field] = st.selectbox(field, ['Month-to-month', 'One year', 'Two year']) if field == 'Contract' else st.selectbox(field, ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']) if field == 'PaymentMethod' else st.selectbox(field, ['No', 'Yes', 'No internet service'])

    return user_input_values

def main():
    user_input_values = get_user_input()
    
    if st.button('Predict', key='predict_button'):
        # Map values to categorical features
        categorical_values = [map_input_to_values(user_input_values[field], field) for field in user_input_values if field not in ['Tenure', 'MonthlyCharges', 'TotalCharges']]
        
        
        user_inputs = list(user_input_values.values()) + categorical_values

        # Reshape the input data to match the expected shape
        user_inputs_reshaped = np.array(user_inputs).reshape(1, -1)

        # Scale the input data using the loaded scaler
        scaled_inputs = scaler.transform(user_inputs_reshaped)

        output = predict_churn(scaled_inputs)
        prediction = "No" if output[0][0] < 0.5 else "Yes"
        st.success(f"The Predicted Customer Churn is {prediction}")

if __name__ == '_main_':
    main()
