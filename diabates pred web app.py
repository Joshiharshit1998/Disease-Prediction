# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 22:52:27 2024

@author: hp
"""

import  numpy as np
import pickle
import streamlit as st

#loading the model
loaded_model = pickle.load(open('D:/Data Science lib/Disease Prediction/trained_model.sav', 'rb'))

#Function for Prediction

def diabetes_prediction(input_data):
    input_data = (5,166,72,19,175,25.8,0.587,51)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  

def main():
    
    #Title
    st.title('Diabetes prediction web app')
    
    #getting input data from user
    
    Pregnancies=st.text_input('Number of pregnancies')
    Glucose=st.text_input('Glucose level')
    BloodPressure=st.text_input('BP value')
    SkinThickness=st.text_input('SkinThickness value')
    Insulin=st.text_input('Insulin value')
    BMI=st.text_input('BMI value')
    DiabetsPedigreevalue=st.text_input('Diabets Pedigree Functon value')
    Age=st.text_input('Age of person')
    
    
    #code for Prediction
    diagnosis=''
    
    #creating button for prediction
    if st.button('Diabetes Test result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,
                                       SkinThickness,Insulin,BMI,DiabetsPedigreevalue,
                                       Age])
    st.success(diagnosis)
    
    
if __name__=='__main__':
    main()
    

