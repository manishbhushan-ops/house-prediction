import pandas as pd
import streamlit as st
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =========================
# TITLE
# =========================
st.markdown(
    "<h1 style='text-align: center; color: #0f172a; font-size: 48px;'>🏠 House Price Predictor</h1>",
    unsafe_allow_html=True
)

st.caption("Built by Manish Kumar | Machine Learning Project")
st.write("This app predicts house prices using AI based on user inputs.")

# =========================
# CSS STYLING
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #dbeafe, #f8fafc);
    color: #0f172a !important;
}

/* All text */
html, body, [class*="css"] {
    color: #0f172a !important;
}

/* Input labels */
label {
    color: #0f172a !important;
    font-weight: 600;
}

/* Subheaders */
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {
    color: #0f172a !important;
}

}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* 🎯 Target ONLY form submit (Predict button) */
div[data-testid="stFormSubmitButton"] button {
    background-color: #2563eb;
    color: white !important;
    border-radius: 10px;
    font-weight: bold;
    padding: 0.5em 1.5em;
}

/* Hover effect */
div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #1d4ed8;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)



# =========================
# LOAD + TRAIN (CACHED)
# =========================
@st.cache_resource
def load_and_train():
    try:
        file_path = os.path.join(os.path.dirname(__file__), "kc_house_data.csv")
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    features = ['sqft_living15','bedrooms','bathrooms','floors',
                'sqft_lot15', 'sqft_living', 'sqft_lot']

    X = df[features]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, scaler, X_test, y_test, features

model, scaler, X_test, y_test, feature_names = load_and_train()

# =========================
# FORM UI
# =========================
st.markdown("<h2 style='color:#1e293b;'>Enter House Details</h2>", unsafe_allow_html=True)



with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        sqft_living15 = st.number_input("sqft_living15", value=1500)
        bedrooms = st.number_input("bedrooms", value=3)
        bathrooms = st.number_input("bathrooms", value=2)

    with col2:
        floors = st.number_input("floors", value=1)
        sqft_lot15 = st.number_input("sqft_lot15", value=5000)
        sqft_living = st.number_input("sqft_living", value=1800)
        sqft_lot = st.number_input("sqft_lot", value=6000)

    submit = st.form_submit_button("Predict")

# =========================
# PREDICTION
# =========================
if submit:

    if bedrooms <= 0 or bathrooms <= 0 or floors <= 0:
        st.error("Bedrooms, bathrooms, and floors must be greater than 0")
    else:
        new_data = pd.DataFrame({
            'sqft_living15': [sqft_living15],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'floors': [floors],
            'sqft_lot15': [sqft_lot15],
            'sqft_living': [sqft_living],
            'sqft_lot': [sqft_lot]
        })

        new_data_scaled = scaler.transform(new_data)
        prediction = model.predict(new_data_scaled)

        st.success(f"💰 Estimated House Price: ₹{int(prediction[0]):,}")

        # =========================
        # EVALUATION
        # =========================
        pred_test = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred_test)
        r2 = r2_score(y_test, pred_test)

        st.markdown("<h3 style='color:#1e293b;'>📊 Model Performance</h3>", unsafe_allow_html=True)

        st.write(f"MAE: {mae:,.2f}")
        st.write(f"R² Score: {r2:.4f}")

        # =========================
        # FEATURE IMPORTANCE
        # =========================
        st.markdown("<h3 style='color:#1e293b;'>📌 Feature Importance</h3>", unsafe_allow_html=True)

        coef = model.coef_.flatten()

        for col, value in zip(feature_names, coef):
            impact = "↑ increases price" if value > 0 else "↓ decreases price"
            st.write(f"{col}: {value:.2f} ({impact})")

        # =========================
        # ACTUAL VS PREDICTED
        # =========================
        st.markdown("<h3 style='color:#1e293b;'>📊 Actual vs Predicted</h3>", unsafe_allow_html=True)

        fig, ax = plt.subplots()
        ax.scatter(y_test, pred_test)

        # 🔥 perfect prediction line
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                color='red')

        ax.set_xlabel("Actual Prices")
        ax.set_ylabel("Predicted Prices")

        st.pyplot(fig)

        # =========================
        # RESIDUAL PLOT
        # =========================
        st.markdown("<h3 style='color:#1e293b;'>📉 Error Distribution</h3>", unsafe_allow_html=True)

        residuals = y_test - pred_test

        fig2, ax2 = plt.subplots()
        ax2.hist(residuals, bins=30)
        ax2.set_xlabel("Error")
        ax2.set_ylabel("Frequency")

        st.pyplot(fig2)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Made with ❤️ using Streamlit & Scikit-learn")