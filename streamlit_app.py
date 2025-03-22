import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load saved artifacts
def load_artifacts():
    # Load scalers
    with open("standard_scaler.pkl", "rb") as f:
        standard_scaler = pickle.load(f)
    
    with open("robust_scaler.pkl", "rb") as f:
        robust_scaler = pickle.load(f)
    
    # Load fine-tuned model
    with open("fine_tuned_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Load encoders
    with open("encoded_target_variable.pkl", "rb") as f:
        target_mapping = pickle.load(f)
    
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    
    return standard_scaler, robust_scaler, model, target_mapping, label_encoders

# Preprocess user input
def preprocess_input(user_input, label_encoders, standard_scaler, robust_scaler):
    # Convert categorical inputs using label encoders
    for col, le in label_encoders.items():
        if col in user_input:
            user_input[col] = le.transform([user_input[col]])[0]
    
    # Scale numerical inputs
    scaled_features = robust_scaler.transform(pd.DataFrame(user_input, index=[0]))
    user_input_scaled = pd.DataFrame(scaled_features, columns=user_input.keys())
    
    # Apply standard scaling to specific columns if needed
    standard_scaling_columns = ["Height"]
    if standard_scaling_columns:
        user_input_scaled[standard_scaling_columns] = standard_scaler.transform(user_input_scaled[standard_scaling_columns])
    
    return user_input_scaled

# Main Streamlit App
def main():
    st.title("Machine Learning App")
    st.write("This app will predict your obesity level!")
    
    # Load artifacts
    standard_scaler, robust_scaler, model, target_mapping, label_encoders = load_artifacts()
    
    # 1. Show Raw Data
    st.header("Raw Data")
    st.subheader("Input Features (X)")
    st.write(input_df.head())
    
    st.subheader("Target Variable (y)")
    st.write(output_df.head())
    
    # 2. Data Visualization
    st.header("Data Visualization")
    fig, ax = plt.subplots()
    scatter = ax.scatter(input_df["Height"], input_df["Weight"], c=output_df, cmap="viridis")
    ax.set_xlabel("Height")
    ax.set_ylabel("Weight")
    ax.set_title("Height vs Weight by Obesity Level")
    plt.colorbar(scatter, label="Obesity Level")
    st.pyplot(fig)
    
    # 3. User Input
    st.header("User Input")
    user_input = {}
    
    # Numerical Inputs (using st.slider)
    numerical_cols = {
        "Age": (14, 61),
        "Height": (1.45, 1.98),
        "Weight": (39, 173),
        "FCVC": (1, 3),
        "NCP": (1, 4),
        "CH2O": (1, 3),
        "FAF": (0, 3),
        "TUE": (0, 2),
    }
    
    for col, (min_val, max_val) in numerical_cols.items():
        user_input[col] = st.slider(col, min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
    
    # Categorical Inputs (using st.selectbox)
    categorical_cols = {
        "Gender": ["Male", "Female"],
        "family_history_with_overweight": ["yes", "no"],
        "FAVC": ["Sometimes", "Always"],
        "CAEC": ["Sometimes", "Always"],
        "SMOKE": ["yes", "no"],
        "SCC": ["yes", "no"],
        "CALC": ["Sometimes", "Always"],
        "MTRANS": ["Public_Transportation", "Motorbike", "Walking", "Bike", "Car"],
    }
    
    for col, options in categorical_cols.items():
        user_input[col] = st.selectbox(col, options)
    
    # 5. Display User-Inputted Data
    st.header("Data Input by User")
    user_input_df = pd.DataFrame([user_input])
    st.table(user_input_df)
    
    # 6. Preprocess User Input
    user_input_scaled = preprocess_input(user_input, label_encoders, standard_scaler, robust_scaler)
    
    # 7. Predict Probabilities
    st.header("Obesity Prediction")
    probabilities = model.predict_proba(user_input_scaled)
    prob_df = pd.DataFrame(probabilities, columns=target_mapping.keys()).T
    prob_df.columns = ["Probability"]
    st.table(prob_df)
    
    # 8. Final Prediction
    prediction = model.predict(user_input_scaled)[0]
    st.write(f"The predicted output is: {prediction} ({target_mapping[prediction]})")

if __name__ == "__main__":
    main()
