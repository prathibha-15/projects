import numpy as np
import streamlit as st
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('deep_neural_network_model.h5')

# Provided mean and standard deviation values
mean_values = np.array([
    48.02884615, 0.63076923, 0.49615385, 0.44807692, 0.41730769, 0.58653846,
    0.45576923, 0.22307692, 0.44807692, 0.48653846, 0.24230769, 0.45961538,
    0.43076923, 0.375, 0.34423077, 0.16923077
])

std_values = np.array([
    12.13977627, 0.48259653, 0.49998521, 0.49729669, 0.49311457, 0.49245415,
    0.4980398, 0.41630951, 0.49729669, 0.49981875, 0.42847949, 0.49836641,
    0.49518391, 0.48412292, 0.47511677, 0.37495562
])

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🩺 Diabetes Risk Prediction App")
st.write("""
This application predicts the risk of diabetes based on various health parameters. 
Please provide the following information to get your risk assessment.
""")

# Collect user inputs
with st.form("prediction_form"):
    st.header("Patient Information")
    
    age = st.slider("Age", 10, 80, 30)
    
    gender = st.selectbox("Gender", ["Female", "Male"])
    gender = 0 if gender == "Female" else 1
    
    polyuria = st.selectbox("Polyuria (Excessive Urination)", ["No", "Yes"])
    polyuria = 0 if polyuria == "No" else 1
    
    polydipsia = st.selectbox("Polydipsia (Excessive Thirst)", ["No", "Yes"])
    polydipsia = 0 if polydipsia == "No" else 1
    
    sudden_weight_loss = st.selectbox("Sudden Weight Loss", ["No", "Yes"])
    sudden_weight_loss = 0 if sudden_weight_loss == "No" else 1
    
    weakness = st.selectbox("Weakness", ["No", "Yes"])
    weakness = 0 if weakness == "No" else 1
    
    polyphagia = st.selectbox("Polyphagia (Excessive Hunger)", ["No", "Yes"])
    polyphagia = 0 if polyphagia == "No" else 1
    
    genital_thrush = st.selectbox("Genital Thrush", ["No", "Yes"])
    genital_thrush = 0 if genital_thrush == "No" else 1
    
    visual_blurring = st.selectbox("Visual Blurring", ["No", "Yes"])
    visual_blurring = 0 if visual_blurring == "No" else 1
    
    itching = st.selectbox("Itching", ["No", "Yes"])
    itching = 0 if itching == "No" else 1
    
    irritability = st.selectbox("Irritability", ["No", "Yes"])
    irritability = 0 if irritability == "No" else 1
    
    delayed_healing = st.selectbox("Delayed Healing", ["No", "Yes"])
    delayed_healing = 0 if delayed_healing == "No" else 1
    
    partial_paresis = st.selectbox("Partial Paresis", ["No", "Yes"])
    partial_paresis = 0 if partial_paresis == "No" else 1
    
    muscle_stiffness = st.selectbox("Muscle Stiffness", ["No", "Yes"])
    muscle_stiffness = 0 if muscle_stiffness == "No" else 1
    
    alopecia = st.selectbox("Alopecia (Hair Loss)", ["No", "Yes"])
    alopecia = 0 if alopecia == "No" else 1
    
    obesity = st.selectbox("Obesity", ["No", "Yes"])
    obesity = 0 if obesity == "No" else 1
    
    # Submit button
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    # Collect all features into a single array
    input_features = np.array([
        age, gender, polyuria, polydipsia, sudden_weight_loss, weakness,
        polyphagia, genital_thrush, visual_blurring, itching, irritability,
        delayed_healing, partial_paresis, muscle_stiffness, alopecia, obesity
    ])
    
    # Normalize the input features using the provided mean and std values
    normalized_features = (input_features - mean_values) / std_values
    normalized_features = normalized_features.reshape(1, -1)  # Reshape to 2D array with one row
    
    # Make prediction using the normalized input
    prediction = model.predict(normalized_features)[0][0]
    
    # Determine risk level
    risk_threshold = 0.5  # You can adjust this threshold based on model performance
    risk = "High Risk" if prediction >= risk_threshold else "Low Risk"
    confidence_percentage = prediction * 100 if prediction <= 1 else 100
    
    # Display the result
    st.subheader("Prediction Result")
    
    if risk == "High Risk":
        st.error(f"**{risk}** of Diabetes")
    else:
        st.success(f"**{risk}** of Diabetes")
    
    st.write(f"**Prediction Confidence:** {confidence_percentage:.2f}%")
    
    # Additional information
    st.write("""
    **Note:** This prediction is based on machine learning model outputs and should not replace professional medical advice. 
    Consult with healthcare professionals for accurate diagnosis and treatment.
    """)
