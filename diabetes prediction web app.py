# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 20:41:11 2022

@author: Darshan Ajila
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

loaded_model = pickle.load(open('E:/Projects/Diabetes Prediction/trained_model.sav', 'rb'))

diabetes_dataset = pd.read_csv('E:/Projects/Diabetes Prediction/diabetes.csv')

X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(X)
standard_data = scaler.transform(X)

def diabetes_prediction(input_data):
    
    input_np_data = np.asarray(input_data)
    input_data_re = input_np_data.reshape(1,-1)

    std_data = scaler.transform(input_data_re)
    print(std_data)

    prediction = loaded_model.predict(std_data)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is non-Diabetes'
    else:
      return 'The Person is Diabetes'
  
def main():
    st.title('Diabetes Prediction')
    
    Preganics = st.text_input('Number of Pregenics')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age')
    
    diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Preganics, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age ])
        
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()