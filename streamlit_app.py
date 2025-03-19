import streamlit as st
import numpy as np
import joblib

model = joblib.load('car_price_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.title("Car Price Prediction System")
st.sidebar.header("Enter Car Details")

car_name = st.sidebar.text_input("Enter Car Name (Exact Name)")
year = st.sidebar.number_input("Year of Purchase", min_value=2000, max_value=2025, step=1)
car_age = 2025 - year
kms_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500000, step=500)
fuel_type = st.sidebar.selectbox("Fuel Type", ("Petrol", "Diesel", "CNG", "LPG", "Electric"))
fuel_type_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4}
fuel_type = fuel_type_mapping[fuel_type]
seller_type = st.sidebar.selectbox("Seller Type", ("Dealer", "Individual", "Trustmark Dealer"))
seller_type_mapping = {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}
seller_type = seller_type_mapping[seller_type]
transmission = st.sidebar.selectbox("Transmission Type", ("Manual", "Automatic"))
transmission_mapping = {'Manual': 0, 'Automatic': 1}
transmission = transmission_mapping[transmission]
owner_type = st.sidebar.selectbox("Owner Type", ("First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"))
owner_mapping = {'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2, 'Fourth & Above Owner': 3, 'Test Drive Car': 4}
owner_type = owner_mapping[owner_type]

if st.sidebar.button("Predict Price"):
    try:
        car_name_encoded = label_encoder.transform([car_name])[0]
    except ValueError:
        st.error("❌ Car name not recognized! Please enter a valid car name.")
        car_name_encoded = -1

    if car_name_encoded != -1:
        input_features = np.array([[car_name_encoded, car_age, kms_driven, fuel_type, seller_type, transmission, owner_type]])
        input_features_scaled = scaler.transform(input_features)
        prediction = model.predict(input_features_scaled)
        predicted_price = round(prediction[0], 2)
        st.success(f"💰 Estimated Selling Price: ₹ {predicted_price/13:.2f} lakhs")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Uzair Baig")
