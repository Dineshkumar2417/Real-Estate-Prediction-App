import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt  # Naya import graphs ke liye!

# Page Configuration (Optional but layout accha lagta hai)
st.set_page_config(page_title="Real Estate Predictor", page_icon="🏠", layout="wide")

# Page Title
st.title("🏠 My Real Estate Price Predictor")
st.write("This is a project I built to practice Linear Regression, Streamlit, and Data Visualization.")

# Sidebar for user interaction
st.sidebar.title("Settings")
name = st.sidebar.text_input("What is your name?")
if name:
    st.sidebar.write(f"Welcome, {name}!")

# Load the data
df = pd.read_csv("data1.csv")

# Show the data and Insights
if st.checkbox("Show the dataset and Insights"):
    st.write("### Dataset Preview")
    st.write(df)
    
    # Naya Graph: Area vs Price
    st.write("### Area vs Price Trend")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(df['area'], df['price'], color='#5c93fa', alpha=0.6)
    ax.set_xlabel("Area (sq ft)")
    ax.set_ylabel("Price ($)")
    ax.set_title("Market Trend: How Area affects the Price")
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

# --- MODEL TRAINING ---
rg = linear_model.LinearRegression()
X = df[['area', 'rooms', 'person']]
y = df['price']
rg.fit(X, y)

st.divider()

# --- USER INPUT ---
st.subheader("Enter Details to Predict Price")

# 3 Columns aur Sliders UI ko aur better banane ke liye
col1, col2, col3 = st.columns(3)
with col1:
    area = st.slider("Area (sq ft):", min_value=100, max_value=10000, value=1000, step=50)
with col2:
    rooms = st.slider("Number of Rooms:", min_value=1, max_value=10, value=2)
with col3:
    person = st.slider("Number of Persons:", min_value=1, max_value=15, value=1)

# --- PREDICTION ---
# Button ko wide kiya hai
submit = st.button("Predict Price", use_container_width=True)

if submit:
    # Prepare the input for the model
    data = np.array([[area, rooms, person]])
    prediction = rg.predict(data)[0]

    st.subheader("Result")
    
    # Normal text ki jagah st.metric use kiya
    st.metric(label="Estimated Property Price", value=f"${prediction:,.2f}")
    st.balloons() # Thoda celebration!
    
    # Naya Graph: User ki property ko market data mein highlight karna
    st.write("### Market Comparison")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    
    # Pura dataset plot kiya
    ax2.scatter(df['area'], df['price'], color='#5c93fa', alpha=0.4, label='Market Data')
    
    # User ki predicted property highlight ki (Red Star)
    ax2.scatter(area, prediction, color='red', s=200, marker='*', label='Your Estimated Property', edgecolor='black')
    
    ax2.set_xlabel("Area (sq ft)")
    ax2.set_ylabel("Price ($)")
    ax2.set_title("Where your property stands in the market")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig2)
    
else:
    st.info("Adjust the sliders and click Predict to see the magic.")