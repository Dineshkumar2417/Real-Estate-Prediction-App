import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model

# Page Title
st.title("🏠 My Real Estate Price Predictor")
st.write("This is a project I built to practice Linear Regression and Streamlit.")

# Sidebar for user interaction
st.sidebar.title("Settings")
name = st.sidebar.text_input("What is your name?")
if name:
    st.sidebar.write(f"Welcome, {name}!")

# Load the data
# Make sure 'data1.csv' is in the same folder as this script!
df = pd.read_csv("data1.csv")

# Show the data (Optional but looks cool for a project)
if st.checkbox("Show the dataset I used"):
    st.write(df)

# --- MODEL TRAINING ---
# I am training the model directly on the dataset
rg = linear_model.LinearRegression()
X = df[['area', 'rooms', 'person']]
y = df['price']
rg.fit(X, y)

st.divider()

# --- USER INPUT ---
st.subheader("Enter Details to Predict Price")

col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (sq ft):", value=1000)
    rooms = st.number_input("Number of Rooms:", value=2)
    person = st.number_input("Number of Persons:", value=1)

# --- PREDICTION ---
submit = st.button("Predict Price")

if submit:
    # Prepare the input for the model
    data = np.array([[area, rooms, person]])
    prediction = rg.predict(data)[0]

    st.subheader("Result")
    st.success(f"The estimated price is: ${prediction:,.2f}")
else:
    st.info("Fill in the details and click Predict.")