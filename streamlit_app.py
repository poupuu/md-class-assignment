import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import joblib

#load .pkl
def load_artifacts():
    #scalers
    with open("standard_scaler.pkl", "rb") as f:
        standard_scaler = pickle.load(f)
    
    with open("robust_scaler.pkl", "rb") as f:
        robust_scaler = pickle.load(f)
    
    #model
    with open("fine_tuned_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    #encoders
    with open("encoded_target_variable.pkl", "rb") as f:
        target_mapping = pickle.load(f)
    
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    
    return standard_scaler, robust_scaler, model, target_mapping, label_encoders

#load user input
def load_input(user_input):
    data = [user_input]
    df = pd.DataFrame(data, columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'])
    return df

#preprocess
def preprocess_input(user_input, label_encoders, standard_scaler, robust_scaler):
    #convert
    for col, le in label_encoders.items():
        if col in user_input:
            user_input[col] = le.transform([user_input[col]])[0]
    
    #scaling
    scaled_features = robust_scaler.transform(pd.DataFrame(user_input, index=[0]))
    user_input_scaled = pd.DataFrame(scaled_features, columns=user_input.keys())
    
    #standard scaling
    standard_scaling_columns = ["Height"]
    if standard_scaling_columns:
        user_input_scaled[standard_scaling_columns] = standard_scaler.transform(user_input_scaled[standard_scaling_columns])
    
    return user_input_scaled

#main
def main():
    st.title("Machine Learning App")
    st.info("This app will predict your obesity level!")
    
    #load
    standard_scaler, robust_scaler, model, target_mapping, label_encoders = load_artifacts()
    
    #raw data
    with st.expander("**Data**"):
        st.write("This is raw data")
        df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
        df

        st.write("**X**")
        input_df = df.drop("NObeyesdad", axis=1)
        input_df
        
        st.write("y")
        output_df = df["NObeyesdad"]
        output_df

    #data visualization
    with st.expander('**Data Visualization**'):
        st.scatter_chart(data=df, x = 'Height', y = 'Weight', color='NObeyesdad')

    #input data
    #numerical
    Age = st.slider('Age', min_value = 14, max_value = 61, value = 24)
    Height = st.slider('Height', min_value = 1.45, max_value = 1.98, value = 1.7)
    Weight = st.slider('Weight', min_value = 39, max_value = 173, value = 86)
    FCVC = st.slider('FCVC', min_value = 1, max_value = 3, value = 2)
    NCP = st.slider('NCP', min_value = 1, max_value = 4, value = 3)
    CH2O = st.slider('CH2O', min_value = 1, max_value = 3, value = 2)
    FAF = st.slider('FAF', min_value = 0, max_value = 3, value = 1)
    TUE = st.slider('TUE', min_value = 0, max_value = 2, value = 1)

    #categorical
    Gender = st.selectbox('Gender', ('Male', 'Female'))
    family_history_with_overweight = st.selectbox('Family history with overweight', ('yes', 'no'))
    FAVC = st.selectbox('FAVC', ('yes', 'no'))
    CAEC = st.selectbox('CAEC', ('Sometimes', 'Frequently', 'Always', 'no'))
    SMOKE = st.selectbox('SMOKE', ('yes', 'no'))
    SCC = st.selectbox('SCC', ('yes', 'no'))
    CALC = st.selectbox('CALC', ('Sometimes', 'no', 'Frequently', 'Always'))
    MTRANS = st.selectbox('MTRANS', ('Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'))
    
    # Input Data for Program
    user_input = [Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS]
    df = load_input(user_input)

    st.write('Data input by user')
    df
    
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
    
    # for col, (min_val, max_val) in numerical_cols.items():
    #     user_input[col] = st.slider(col, min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
    
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

    # df = 
    # df = sca
    
    # for col, options in categorical_cols.items():
    #     user_input[col] = st.selectbox(col, options)
    
    #preprocess user input
    user_input_scaled = preprocess_input(user_input, label_encoders, standard_scaler, robust_scaler)
    
    #predict probabilities
    st.write("Obesity Prediction")
    probabilities = model.predict_proba(user_input_scaled)
    prob_df = pd.DataFrame(probabilities, columns=target_mapping.keys()).T
    prob_df.columns = ["Probability"]
    st.table(prob_df)
    
    #prediction
    prediction = model.predict(user_input_scaled)[0]
    st.write(f"The predicted output is: ",prediction)

if __name__ == "__main__":
    main()

