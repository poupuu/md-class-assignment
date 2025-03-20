import streamlit as st
import joblib
import pandas as pd

model = joblib.load('trained_model.pkl')
loaded_encoder = joblib.load('encoder.pkl')
loaded_scaler = joblib.load('scaler.pkl')

def input_to_df(input):
  data = [input]
  df = pd.DataFrame(data, columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'])
  return df

def encode(df):
  for column in df.columns:
    if df[column].dtype == "object":
      df[column] = loaded_encoder.fit_transform(df[column])
  return df

def normalize(df):
  df = loaded_scaler.transform(df)
  return df

def predict_with_model(model, user_input):
  prediction = model.predict(user_input)
  return prediction[0]

def main():
  st.title('Machine Learning App')
  
  st.info('This app will predict your obesity level!')
  
  # Raw Data
  with st.expander('**Data**'):
    st.write('This is a raw data')
    df = pd.read_csv('https://raw.githubusercontent.com/JeffreyJuinior/dp-machinelearning/refs/heads/master/ObesityDataSet_raw_and_data_sinthetic.csv')
    df
    st.write('**X**')
    X = df.drop('NObeyesdad',axis=1)
    X
    st.write('**y**')
    y = df['NObeyesdad']
    y
  
  # Visualization
  with st.expander('**Data Visualization**'):
    st.scatter_chart(data=df, x = 'Height', y = 'Weight', color='NObeyesdad')
  
  # Input Data bu User
  Age = st.slider('Age', min_value = 14, max_value = 61, value = 24)
  Height = st.slider('Height', min_value = 1.45, max_value = 1.98, value = 1.7)
  Weight = st.slider('Weight', min_value = 39, max_value = 173, value = 86)
  FCVC = st.slider('FCVC', min_value = 1, max_value = 3, value = 2)
  NCP = st.slider('NCP', min_value = 1, max_value = 4, value = 3)
  CH2O = st.slider('CH2O', min_value = 1, max_value = 3, value = 2)
  FAF = st.slider('FAF', min_value = 0, max_value = 3, value = 1)
  TUE = st.slider('TUE', min_value = 0, max_value = 2, value = 1)
  
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
  df = input_to_df(user_input)

  st.write('Data input by user')
  df

  df = encode(df)
  df = normalize(df)
  prediction = predict_with_model(model, df)
  
  prediction_proba = model.predict_proba(df)
  df_prediction_proba = pd.DataFrame(prediction_proba)
  df_prediction_proba.columns = ['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 'Overweight Level II', 'Obesity Type I', 'Obesity Type II', 'Obesity Type III']
  df_prediction_proba.rename(columns={0: 'Insufficient Weight', 
                                      1:'Normal Weight', 
                                      2: 'Overweight Level I', 
                                      3: 'Overweight Level II', 
                                      4:'Obesity Type I', 
                                      5:'Obesity Type II', 
                                      6: 'Obesity Type III'})

  st.write('Obesity Prediction')
  df_prediction_proba
  st.write('The predicted output is: ',prediction) 
  

if __name__ == "__main__":
  main()
