import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and preprocessor
model = joblib.load("artifacts/model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

# Define the main function
def main():
    # Set page title and layout
    st.set_page_config(page_title="Accident Severity Prediction App", layout="wide")
    st.title("Accident Severity Prediction App")

    # Define dropdown options
    options_day = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    options_driving_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']
    options_junction_type = ['Y Shape', 'No junction', 'Crossing', 'Other', 'Unknown', 'O Shape', 'T Shape', 'X Shape']
    options_road_conditions = ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep']
    options_sex = ['Male', 'Female', 'Unknown']
    options_age = ['na', '18-30', '31-50', 'Under 18', 'Over 51', '5']
    options_severity = ['3', 'na', '2', '1']
    options_fitness = ['Normal', 'NormalNormal', 'Blind', 'Deaf', 'Other']

    # Define the form
    with st.form("accident_severity_form"):
        # Add form inputs
        st.subheader("Please enter the following inputs:")
        day = st.selectbox("Day of the week", options=options_day)
        driving_exp = st.selectbox("Driving experience", options=options_driving_exp)
        junction_type = st.selectbox("Junction type", options=options_junction_type)
        road_conditions = st.selectbox("Road conditions", options=options_road_conditions)
        num_of_vehicles = st.slider("Number of vehicles involved", 1, 7, 1)
        casualty = st.slider("Number of casualties", 1, 8, 1)
        sex = st.selectbox("Sex of casualty", options=options_sex)
        age = st.selectbox("Age of casualty", options=options_age)
        severity = st.selectbox("Severity of accident", options=options_severity)
        fitness = st.selectbox("Fitness of casualty", options=options_fitness)

        # Add submit button
        submit_button = st.form_submit_button(label='Predict')

    # If submit button is clicked
    if submit_button:
        # Encode the input using the preprocessor
        input_array = np.array([day, driving_exp, junction_type, road_conditions, num_of_vehicles, casualty, sex, age, severity, fitness], ndmin=2)
        encoded_array = list(preprocessor.transform(input_array).ravel())
        num_array = [num_of_vehicles, casualty]
        pred_array = np.array(num_array + encoded_array).reshape(1, -1)

        # Make the prediction
        prediction = model.predict(pred_array)

        # Show the prediction
        st.subheader("Prediction:")
        st.write("The predicted severity of the accident is:", prediction[0])

# Run the main function
if __name__ == '__main__':
    main()
