import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import joblib

# Load saved artifacts
target_encoder = joblib.load("encoded_target_variable.pkl")
label_encoders = joblib.load("label_encoders.pkl")
standard_scaler = joblib.load("standard_scaler.pkl")
robust_scaler = joblib.load("robust_scaler.pkl")
model = joblib.load("fine_tuned_model.pkl")

# Convert user input to DataFrame
def input_user_to_df(user_input):
    columns = [
        'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 
        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 
        'CALC', 'MTRANS'
    ]
    df = pd.DataFrame([user_input], columns=columns)
    return df

# Encode categorical features
def feature_encode(df, label_encoders):
    for column in df.columns:
        if column in label_encoders:  # Check if the column has a corresponding encoder
            try:
                df[column] = label_encoders[column].transform(df[column])
            except Exception as e:
                st.error(f"Error encoding column '{column}': {e}")
                raise
    return df

# Apply standard scaling to "Height"
def standard_scaling(df, standard_scaler):
    df["Height"] = standard_scaler.transform(df[["Height"]])
    return df

# Apply robust scaling to numerical columns (except "Height")
def robust_scaling(df, robust_scaler):
    numerical_columns = ["Age", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    for column in numerical_columns:
        df[column] = robust_scaler.transform(df[[column]])
    return df

# Predict probabilities and final class
def predict_model(model, user_input_scaled):
    probabilities = model.predict_proba(user_input_scaled)
    prediction = model.predict(user_input_scaled)[0]
    return probabilities, prediction

# Main Streamlit App
def main():
    st.title("Obesity Level Prediction App")
    st.info("This app predicts your obesity level based on your inputs!")

    # Raw Data Section
    with st.expander("**Raw Data**"):
        st.write("This is the raw dataset used for training the model.")
        df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
        st.write(df.head())

        st.write("**Features (X)**")
        input_df = df.drop("NObeyesdad", axis=1)
        st.write(input_df.head())

        st.write("**Target Variable (y)**")
        output_df = df["NObeyesdad"]
        st.write(output_df.head())

    # Data Visualization Section
    with st.expander("**Data Visualization**"):
        fig, ax = plt.subplots()
        scatter = ax.scatter(df["Height"], df["Weight"], c=output_df.astype('category').cat.codes, cmap="viridis")
        ax.set_xlabel("Height")
        ax.set_ylabel("Weight")
        ax.set_title("Height vs Weight by Obesity Level")
        plt.colorbar(scatter, label="Obesity Level")
        st.pyplot(fig)

    # User Input Section
    st.header("Enter Your Information")
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
        "FAVC": ["yes", "no"],
        "CAEC": ["Sometimes", "Frequently", "Always", "no"],
        "SMOKE": ["yes", "no"],
        "SCC": ["yes", "no"],
        "CALC": ["Sometimes", "no", "Frequently", "Always"],
        "MTRANS": ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"],
    }

    for col, options in categorical_cols.items():
        user_input[col] = st.selectbox(col, options)

    # Display User-Inputted Data
    st.header("Your Input Data")
    user_input_df = input_user_to_df(user_input)
    st.table(user_input_df)

    # Preprocess User Input
    # Step 1: Encode categorical features
    user_input_encoded = feature_encode(user_input_df, label_encoders)

    # Step 2: Scale numerical features
    user_input_scaled = standard_scaling(user_input_encoded, standard_scaler)
    user_input_scaled = robust_scaling(user_input_scaled, robust_scaler)

    # Predict Probabilities and Final Class
    probabilities, prediction = predict_model(model, user_input_scaled)

    # Display Prediction Results
    st.header("Prediction Results")
    
    # Probabilities Table
    prob_df = pd.DataFrame(probabilities, columns=[
        "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", 
        "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", 
        "Obesity_Type_III"
    ]).T
    prob_df.columns = ["Probability"]
    st.subheader("Class Probabilities")
    st.table(prob_df)

    # Final Prediction
    st.subheader("Final Prediction")
    st.success(f"The predicted obesity level is: **{prediction}**")

if __name__ == "__main__":
    main()
