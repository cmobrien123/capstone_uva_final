import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
# import csv


# SETTING PAGE CONFIG TO Centered MODE
st.set_page_config(layout="wide")

st.title("""
Anomaly Detection of Electrical Signals in a Particle Accelerator
""")

st.write("""
By Colin O'Brien
""")

## Data overview image
Data_overview = Image.open('parameters.png')
st.image(Data_overview, width = 400)



## Encoder image
encoder_overview = Image.open('encoder.png')
st.image(encoder_overview, width = 1200)


## pulling in data
training_df_results = pd.read_csv('Train_data_latent_df.csv', header=0)
test_df_results = pd.read_csv('Test_data_latent_df.csv', header=0)


## Training Data Results:
Training_df_summary = pd.DataFrame({
'Accuracy':92.86,
'Precision':100.00,
'Recall':69.00,
'F1-Score':81.63
}, index=[0])

Training_df_summary

## Training Data Plot
st.write("""
Training Data
""")

fig = px.scatter_3d(pd.DataFrame(training_df_results), x='Dim 0',
                    y='Dim 1', z='Dim 2',
                color='Classification Results',
                    hover_data = ['Fault Type'],
                color_discrete_map={
                'TN': 'blue',
                'TP': 'green',
                'FP': 'orange',
                'FN': 'red'}
                   )
fig.update_layout(title_text='Training Data Latent Space', title_x=0.5)
fig

st.write("""
Test Data
""")
## Test Data Results:
Test_df_summary = pd.DataFrame({
'Accuracy':87.65,
'Precision':86.67,
'Recall':61.90,
'F1-Score':72.22
}, index=[0])

Test_df_summary

fig = px.scatter_3d(pd.DataFrame(test_df_results), x='Dim 0',y='Dim 1',z='Dim 2',
                   color='Classification Results',
                    hover_data = ['Fault Type'],
                color_discrete_map={
                'TN': 'blue',
                'TP': 'green',
                'FP': 'orange',
                'FN': 'red'})
fig.update_layout(title_text='Test Data Latent Space', title_x=0.5)
fig
