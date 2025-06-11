import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")

st.title("🏠 House Price Predictor")
st.write("Upload a CSV file with 'Area' and 'Price' columns to train the model.")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df)

    if 'Area' in df.columns and 'Price' in df.columns:
        X = df[['Area']]
        y = df['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        st.success(f"✅ Model trained! Accuracy: {round(score * 100, 2)}%")

        st.subheader("📈 Prediction Plot")
        fig, ax = plt.subplots()
        sns.scatterplot(x='Area', y='Price', data=df, s=100)
        sns.lineplot(x=df['Area'], y=model.predict(X), color='red')
        st.pyplot(fig)

        st.subheader("🔮 Predict New Price")
        area = st.number_input("Enter Area (sq ft)", min_value=0)
        if area:
            pred_price = model.predict([[area]])[0]
            st.info(f"Predicted Price for {area} sq ft = ₹{round(pred_price, 2)} lakhs")
    else:
        st.error("CSV must contain 'Area' and 'Price' columns.")
else:
    st.info("Upload a CSV file to get started.")
