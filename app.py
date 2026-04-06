import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

@st.cache_data
def load_brands():
    return sorted(pd.read_csv("data/cleaned_cardekho_used_cars.csv")["brand"].unique().tolist())

st.set_page_config(
    page_title='Used Car Price Predictor',
    page_icon='🚗',
    layout='centered'
)

st.title('🚗 Used Car Price Prediction')
st.write('Get instant price estimates for used cars using machine learning')

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("models/catboost_model.cbm")
    return model

try:
    model = load_model()
    st.success("✓ Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.header("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox(
        "Brand",
        load_brands()
    )
    
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    
    ownership = st.selectbox("Ownership", ["First Owner", "Second Owner", "Third Owner"])
    
    insurance = st.selectbox("Insurance Type", ["Comprehensive", "Own Damage", "Zero Dep", "Third Party"])

with col2:
    seats = st.number_input("Seats", min_value=2, max_value=9, value=5, step=1)
    
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)
    
    engine_displacement = st.number_input("Engine Displacement (cc)", min_value=800, max_value=5000, value=1200, step=100)
    
    manufacture_yr = st.number_input("Manufacture Year", min_value=1995, max_value=2026, value=2020, step=1)

st.markdown("---")

if st.button("🔍 Predict Price", type="primary", use_container_width=True):
    if manufacture_yr > 2026:
        st.error("⚠️ Manufacture year cannot be in the future!")
    elif km_driven < 0:
        st.error("⚠️ Kilometers driven cannot be negative!")
    else:
        with st.spinner("Calculating price..."):
            input_df = pd.DataFrame([{
                "brand": brand,
                "insurance": insurance,
                "fuel_type": fuel_type,
                "seats": seats,
                "km_driven": km_driven,
                "ownership": ownership,
                "engine_displacement": float(engine_displacement),
                "transmission": transmission,
                "manufacture_yr": manufacture_yr
            }])
            
            try:
                log_price = model.predict(input_df)
                price = np.exp(log_price)[0]
                
                st.success(f"### Estimated Price: ₹{price:,.0f}")

                lower = max(0, price - 150000)
                upper = price + 150000
                st.info(f"**Estimated Range**: ₹{lower/100000:.2f}L – ₹{upper/100000:.2f}L")

                st.markdown("---")
                st.caption("💡 **Note**: Model predictions within ±₹1.5L ~73% of the time on test data. Actual prices may vary based on car condition, location, and market demand.")
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='color: gray; font-size: 0.9em;'>
        Powered by CatBoost ML Model | R² Score: 0.793 | MAE: ₹146,009
    </p>
</div>
""", unsafe_allow_html=True)
