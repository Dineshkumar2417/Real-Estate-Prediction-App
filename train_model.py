import pandas as pd
from sklearn import linear_model
import joblib

# Load data
df = pd.read_csv("data1.csv")

# Train model
rg = linear_model.LinearRegression()
X = df[['area','rooms','person']]
y = df['price']
rg.fit(X, y)

# SAVE the model to a file
joblib.dump(rg, 'model.pkl')
print("Model saved as model.pkl")