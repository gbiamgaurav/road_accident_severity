## Install dependencies
import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import joblib
import shap
import matplotlib
from IPython import get_ipython
from PIL import Image

## load the encoder and model object
model=joblib.load("artifacts/model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

st.set_page_config(page_title="Accident Severity Prediction App",
                   layout="wide")

# creating option list for dropdown menu
options_day = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
options_driving_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']

options_junction_type = ['Y Shape', 'No junction', 'Crossing', 'Other', 'Unknown', 'O Shape', 'T Shape', 'X Shape']

options_road_conditions = ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep']




# number of vehicles involved: range of 1 to 7
# number of casualties: range of 1 to 8
# hour of the day: range of 0 to 23

options_sex = ['Male', 'Female', 'Unknown']

options_age = ['na', '18-30', '31-50', 'Under 18', 'Over 51', '5']

options_severity = ['3', 'na', '2', '1']

options_fitness = ['Normal', 'NormalNormal', 'Blind', 'Deaf', 'Other']


## feature list

features = ['weekday', 'driving_exp', 'junction_type', 'road_conditions',
       'num_of_vehicles', 'casualty', 'casualty_sex', 'casualty_age',
       'casualty_severity', 'casualty_fitness']


## Give a title to web app using html syntax

st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)

# define a main() function to take inputs from user in form based approch
def app():
       with st.form("road_traffic_severity_form"):
              st.subheader("Pleas enter the following inputs:")
              
              day = st.selectbox("Day of the week: ", options=options_day)
              driving_exp = st.selectbox("Driving exp: ", options=options_driving_exp)
              juntion_type = st.selectbox("Junction Type: ", options=options_junction_type)
              
              road_conditions = st.selectbox("Road conditions: ", options=options_junction_type)
              
              num_of_vehicles = st.select_slider("Number of Vehicles: ", 1, 8, value=0)
              
              casualty = st.select_slider("Casualty: ", 1, 8, value=0)
              
              sex = st.selectbox("Sex: ", options=options_sex)
              
              age = st.selectbox("Age: ", options=options_age)
              
              severity = st.selectbox("Severity: ", options=options_severity)
              
              fitness = st.selectbox("Fitness: ", options=options_fitness)
            
              
              submit = st.form_submit_button("Predict")

# encode using ordinal encoder and predict
       if submit:
              input_array = np.array([day,
                                      driving_exp, juntion_type, road_conditions,
                                      num_of_vehicles, casualty, sex,
                                      age, severity, fitness], ndmin=2)
              
              encoded_arr = list(preprocessor.transform(input_array).ravel())
              
              num_arr = [num_of_vehicles, casualty]
              pred_arr = np.array(num_arr + encoded_arr).reshape(1,-1)              
          
# predict the target from all the input features
              prediction = model.predict(pred_arr)
              
              if prediction == 0:
                     st.write(f"The severity prediction is Fatal Injury")
              elif prediction == 1:
                     st.write(f"The severity prediction is Serious Injury")
              else:
                     st.write(f"The severity prediciton is Slight Injury")
                  
              st.subheader("Explainable AI (XAI) to understand predictions")


## run the main function

if __name__ == "__main__":
    app.run()