import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.markdown(
    "<h1 style='text-align: center; color: #0f172a; font-size: 48px;'>🏠 House Price Predictor</h1>",
    unsafe_allow_html=True
)



st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #dbeafe, #f8fafc);
}

/* Main Title */
h1 {
    color: #0f172a;
    text-align: center;
    font-weight: 700;
}

/* Subheadings */
h2, h3 {
    color: #1e293b;
    font-weight: 600;
}

/* General text */
p {
    color: #334155;
}

/* Button styling */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)



# =========================
# LOAD + TRAIN (CACHED)
# =========================
@st.cache_resource
def load_and_train():
    df = pd.read_csv('kc_house_data.csv')

    X = df[['sqft_living15','bedrooms','bathrooms','floors',
            'sqft_lot15', 'sqft_living', 'sqft_lot']]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, scaler, X_test, y_test, X.columns

model, scaler, X_test, y_test, feature_names = load_and_train()

# =========================
# FORM UI (NEW 🔥)
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
# VALIDATION + PREDICTION
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

        st.success(f"💰 Predicted Price: ₹{int(prediction[0]):,}")

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

        for i, col in enumerate(feature_names):
            impact = "↑ increases price" if coef[i] > 0 else "↓ decreases price"
            st.write(f"{col}: {coef[i]:.2f} ({impact})")

        # =========================
        # PLOT
        # =========================
        st.markdown("<h2 style='color:#1e293b;'>📊 Actual vs Predicted</h2>", unsafe_allow_html=True)

        fig, ax = plt.subplots()
        ax.scatter(y_test, pred_test)
        ax.set_xlabel("Actual Prices")
        ax.set_ylabel("Predicted Prices")

        st.pyplot(fig)

# =========================
# ABOUT
# =========================
st.markdown("---")
st.write("This app predicts house prices using Linear Regression.")