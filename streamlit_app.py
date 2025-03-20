import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model, encoder, scaler, and target encoder
model = joblib.load('trained_model_tuned.pkl')
ordinal_encoder = joblib.load('encode.pkl')
scaler = joblib.load('scaling.pkl')

# Function to convert user input into a DataFrame
def input_to_df(input_data):
    columns = [
        'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 
        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'
    ]
    return pd.DataFrame([input_data], columns=columns)

# Function to encode categorical variables
def encode(df):
    # Define the categorical columns explicitly
    categorical_columns = [
        'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 
        'SMOKE', 'SCC', 'CALC', 'MTRANS'
    ]
    
    # Encode categorical columns using the loaded OrdinalEncoder
    df[categorical_columns] = ordinal_encoder.transform(df[categorical_columns])
    return df

# Function to normalize numerical features
def normalize(df):
    scaled_data = scaler.transform(df)
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
    return df_scaled

# Function to predict using the trained model
def predict_with_model(model, user_input):
    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)
    return prediction[0], prediction_proba[0]

# Main function to run the Streamlit app
def main():
    st.title('Obesity Level Prediction App')
    st.info('This app predicts obesity levels based on user input.')

    # Raw Data Display
    with st.expander('**Raw Data**'):
        st.write("Below is the raw dataset used for training the model.")
        df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
        st.write(df)

    # Data Visualization
    with st.expander('**Data Visualization**'):
        st.write("Visualization of Weight vs Height, colored by Obesity Level.")
        st.scatter_chart(
            data=df,
            x='Height',
            y='Weight',
            color='NObeyesdad'
        )

    # User Input Section
    st.subheader("Input Your Data")
    Age = st.slider('Age', min_value=14, max_value=61, value=24)
    Height = st.slider('Height (m)', min_value=1.45, max_value=1.98, value=1.7)
    Weight = st.slider('Weight (kg)', min_value=39, max_value=173, value=86)
    FCVC = st.slider('Frequency of Consuming Vegetables (FCVC)', min_value=1, max_value=3, value=2)
    NCP = st.slider('Number of Main Meals (NCP)', min_value=1, max_value=4, value=3)
    CH2O = st.slider('Consumption of Water Daily (CH2O)', min_value=1, max_value=3, value=2)
    FAF = st.slider('Physical Activity Frequency (FAF)', min_value=0, max_value=3, value=1)
    TUE = st.slider('Time Using Technology Devices (TUE)', min_value=0, max_value=2, value=1)

    Gender = st.selectbox('Gender', ('Male', 'Female'))
    family_history_with_overweight = st.selectbox('Family History with Overweight', ('yes', 'no'))
    FAVC = st.selectbox('Frequent Consumption of High Caloric Food (FAVC)', ('yes', 'no'))
    CAEC = st.selectbox('Consumption of Food Between Meals (CAEC)', ('Sometimes', 'Frequently', 'Always', 'no'))
    SMOKE = st.selectbox('Smoking Habit (SMOKE)', ('yes', 'no'))
    SCC = st.selectbox('Calories Consumption Monitoring (SCC)', ('yes', 'no'))
    CALC = st.selectbox('Consumption of Alcohol (CALC)', ('Sometimes', 'no', 'Frequently', 'Always'))
    MTRANS = st.selectbox('Transportation Used (MTRANS)', ('Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'))

    # Collect user input into a list
    user_input = [
        Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS
    ]
    df_user_input = input_to_df(user_input)

    # Display user input in table form
    st.subheader("User Input Summary")
    st.write(df_user_input)

    # Encode and Normalize the user input
    df_encoded = encode(df_user_input)
    df_normalized = normalize(df_encoded)

    # Predict using the model
    prediction, prediction_proba = predict_with_model(model, df_normalized)

    # Map the predicted class to its label
    class_labels = {
        0: 'Insufficient Weight',
        1: 'Normal Weight',
        2: 'Overweight Level I',
        3: 'Overweight Level II',
        4: 'Obesity Type I',
        5: 'Obesity Type II',
        6: 'Obesity Type III'
    }

    # Decode the predicted class
    predicted_class_label = class_labels[prediction]

    # Display prediction probabilities for each class
    st.subheader("Prediction Probabilities")
    df_prediction_proba = pd.DataFrame([prediction_proba], columns=class_labels.values())
    st.write(df_prediction_proba)

    # Display the final predicted class
    st.subheader("Final Prediction")
    st.write(f"The predicted obesity level is: **{predicted_class_label}**")

if __name__ == "__main__":
    main()
