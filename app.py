import streamlit as st
import pickle
import numpy as np
import pandas as pd

pipe = pickle.load(open("E:\sunbase\model_pipeline1.pkl",'rb'))

st.set_page_config(page_title='Customer Churn')

st.title('Customer Churn Prediction')

Age_Group= st.selectbox('Age Group', ['Teenager', 'Young Adult', 'Adult', 'Senior'])
Gender = st.selectbox('Gender', ['male', 'female'])
Location = st.selectbox('Location', ['New York', 'Chicago', 'Miami', 'Los Angeles', 'Houston'])

Subscription_Length_Months = st.number_input('Subscription Length Months',min_value=0,max_value=30)
Avg_Monthly_Data_Usage = st.number_input('Avg Monthly Data Usage',min_value=0,max_value=1000)
Monthly_Bill = st.number_input('Monthly Bill',min_value=0,max_value=200)
Total_Usage_GB = st.number_input('Total Usage (GB)',min_value=0,max_value=1000)

# Create a dictionary with user inputs
input_data = {
    'Age_Group': Age_Group,
    'Gender': Gender,
    'Location': Location,
    'Subscription_Length_Months': Subscription_Length_Months,
    'Avg Monthly_Data_Usage': Avg_Monthly_Data_Usage,
    'Monthly_Bill': Monthly_Bill,
    'Total_Usage_GB': Total_Usage_GB
}

# Convert the dictionary to a DataFrame
input_df = pd.DataFrame([input_data])

# Display the user input DataFrame
st.write("User Input Data:")
st.dataframe(input_df)

if st.button('Prediction'):
    # Make a prediction using the pipeline
    prediction = pipe.predict(input_df)
    # Display the prediction result
    if prediction[0] == 1:
       st.title('Yes')
    else:
        st.title("No")




