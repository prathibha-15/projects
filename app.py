import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('deep_neural_network_model.h5')

# Load and preprocess the dataset
df = pd.read_csv('early+stage+diabetes+risk+prediction+dataset.zip')

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split features and target
X = df.drop('class', axis=1)
y = df['class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Streamlit App
st.title('Deep Neural Network Classifier')

# Sidebar for user input features
st.sidebar.header('User Input Features')
def user_input_features():
    features = {}
    for i, col in enumerate(X.columns):
        features[col] = st.sidebar.number_input(col, value=0.0)
    return pd.DataFrame(features, index=[0])

input_data = user_input_features()

# Display input data
st.subheader('User Input Data')
st.write(input_data)

# Scale the input data
input_data_scaled = scaler.transform(input_data)
st.subheader('Scaled Input Data')
st.write(input_data_scaled)

# Make prediction
prediction = model.predict(input_data_scaled)
predicted_class = (prediction > 0.5).astype(int)

st.subheader('Prediction')
st.write('Predicted Probability:', prediction[0][0])
st.write('Predicted Class:', 'Positive' if predicted_class[0][0] == 1 else 'Negative')

# Evaluate the model on test data
st.subheader('Model Evaluation')
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

# Confusion Matrix
st.write('Confusion Matrix:')
cm = confusion_matrix(y_test, y_pred_classes)
st.write(cm)

# Classification Report
st.write('Classification Report:')
report = classification_report(y_test, y_pred_classes, output_dict=True)
st.write(pd.DataFrame(report).transpose())

# ROC Curve
st.subheader('ROC Curve')
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
st.pyplot(plt)

# Precision-Recall Curve
st.subheader('Precision-Recall Curve')
precision, recall, _ = precision_recall_curve(y_test, y_pred)
average_precision = average_precision_score(y_test, y_pred)
plt.figure()
plt.plot(recall, precision, lw=2, label=f'Average precision = {average_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
st.pyplot(plt)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
st.write(f'Accuracy: {accuracy}')
