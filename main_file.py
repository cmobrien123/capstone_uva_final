import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
# import csv


# SETTING PAGE CONFIG TO Centered MODEg
st.set_page_config(layout="wide")



########################
## TO DO'S
########################
## https://docs.streamlit.io/library/api-reference/layout/st.columns

## try to play around with columns this time

########################
## image pull in here
########################

Data_overview = Image.open('parameters.png')


########################
## title
########################

st.title("""
Anomaly Detection of Electrical Signals in a Particle Accelerator
""")

st.header("""
By Colin O'Brien
""")

########################
## intro
########################
blank_l, Centered, blank_r = st.columns(3)
with blank_l:
    st.write(' ')
with Centered:
    st.header("""
    Introduction
    """)
with blank_r:
    st.write(' ')

st.write("""
While studying at the University of Virgina, I took part in a capstone course while I was lucky enough to work with JLab and Oak Ridge facilities (get their full real names) on an anomaly detection problem they were facing. They were trying to use electrical signals from some of the machines in a particle accelerator to predict if the system was about to malfunction. On this page, I want to outline one of the proposed solutions to this problem. The main pieces of the pipeline I used are as follows:

1)	Fourier transformation as a data transformation step
2)	Training an autoencoder
3)	Applying a Radial Support Vector Machine to the latent space of the autoencoder.

Something important to note about the methods I used is that I was not only concerned with building a method that could accurately find abnormal observations, I wanted to also achieve a level of interpretability and ability to visualize the difference between observations.

""")
blank_l, Centered, blank_r = st.columns(3)
with blank_l:
    st.write(' ')
with Centered:
    st.header("""
    Data Overview
    """)
with blank_r:
    st.write(' ')


col_data_write_up, col_data_image = st.columns(2)

with col_data_write_up:
    st.write("""
The data consisted of electrical signals from the HVCM system, one of the key systems in creating the beam of a particle accelerator. The observations fell into one of two categories:

- “Normal”: a signal collected right before the beam was produced in the particle accelerator.
    - I like to think of these as a “normal heartbeat” of the HVCM system.
- “Faults”: Signals collected right before the machine failed to produce the beam, meaning an experiment or other research could not be completed. Once a Fault occurs, the error must be diagnosed, with 14 different types of Faults.
    - I think of these as the “heartbeat right before a heart attack” in the HVCM system.

For each observation, there are 19 different sub-components of the HVCM system, each with a different signal (or heartbeat). To the right is an example for one of the “Normal” observations. The aim is train a method that looks across all 19 parameters and finds differences in these signals in order to make classifications.

""")
with col_data_image:
    st.write("Single Observation Example")
    st.image(Data_overview, width = 400)

################
## pipeline
################

blank_l, Centered, blank_r = st.columns(3)
with blank_l:
    st.write(' ')
with Centered:
    st.header("""
    Building Pipeline
    """)
with blank_r:
    st.write(' ')

blank_l, Centered, blank_r = st.columns(3)
with blank_l:
    st.write(' ')
with Centered:
    st.write("""
    Bag of Symbolic Fourier Approximation Symbols
    """)
with blank_r:
    st.write(' ')

BOSS_write_up, BOSS_image = st.columns(2)

with BOSS_write_up:
    st.write("""
    After splitting the data into training and test data, the first step is applying the Bag of Symbolic Fourier Approximation Symbols (BOSS) transformation. For a given observation, each of the 19 sub-components have a sliding window applied. Each snapshot captured by the sliding window is decomposed by its Fourier coefficient, which is used to assign the snapshot to one of the 8 different words in a 2 letter, 3 index dictionary. As the sliding window moves across the signal, the many snapshots are used to produce a histogram, which can be represented as 1D vector.

    In a less technical perspective, the snapshots are grouped together, with similar snapshots being given the same word/location in the histogram. This process is repeated for all 19 parameters, creating a 19 by 8 matrix for each observation. (Add advantages to doing this here?)

    """)










## BOSS Transformation
encoder_overview = Image.open('BOSS_transformation.png')
st.image(encoder_overview, width = 800)



## Encoder image
encoder_overview = Image.open('even_better_encoder.png')
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
