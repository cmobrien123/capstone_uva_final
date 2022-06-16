import streamlit as st
import pandas as pd
import plotly.express as px
# import csv


# SETTING PAGE CONFIG TO Centered MODE
st.set_page_config(layout="wide")

st.title("""
Anomaly Detection of Electrical Signals in a Particle Accelerator
""")

st.write("""
By Colin O'Brien
""")
## pulling in data
training_df_results = pd.read_csv('Train_data_latent_df.csv', header=0)
test_df_results = pd.read_csv('Test_data_latent_df.csv', header=0)



## plots
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
