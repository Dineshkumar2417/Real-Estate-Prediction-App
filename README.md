# 🏠 Real Estate Price Predictor

Ye project property ki specifications (area, location, etc.) ke basis par uski market price predict karta hai. Isme advanced data cleaning aur feature engineering ka use kiya gaya hai taaki predictions accurate rahein.

## 🚀 Live Demo
Aap is project ka live demo yahan dekh sakte hain:
👉 [**Real Estate Predictor Live**](https://real-estate-prediction-app-ewsjaj2kvwzs4b3tuxnwgu.streamlit.app/)

## ✨ Features
- **Accurate Estimations:** Regression algorithms ka use karke property value predict karta hai.
- **Data Preprocessing:** Handle missing values, outliers, aur categorical data (locations) ka proper treatment.
- **Simple UI:** User-friendly interface jahan sirf basic details daal kar price check ki ja sakti hai.
- **Interactive Visualization:** Price trends aur feature importance ko graphs ke through dikhaya gaya hai.

## 🛠️ Tech Stack
- **Language:** Python
- **Machine Learning:** Scikit-learn (Linear Regression / Random Forest)
- **Data Analysis:** Pandas, NumPy
- **Frontend:** Streamlit
- **Visualization:** Seaborn, Matplotlib

## 📂 Project Structure
- `main.py`: Streamlit web interface ka code.
- `house_model.pkl`: Serialized trained model.
- `EDA_Notebook.ipynb`: Data cleaning aur model selection ka poora process.
- `data/`: Dataset jisme property ki prices aur features hain.

## ⚙️ How to Run Locally
1. Project clone karein.
2. Dependencies install karein: `pip install -r requirements.txt`
3. App start karein: `streamlit run main.py`
