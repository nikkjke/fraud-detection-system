import streamlit as st
import pandas as pd
import joblib

@st.cache_data
def load_fraud_samples():
    return pd.read_csv("fraud_samples.csv")

@st.cache_data
def load_legit_samples():
    return pd.read_csv("legit_samples.csv")

@st.cache_resource()
def load_model():
    return joblib.load("fraud_detection_model.pkl")

@st.cache_resource()
def load_encoder():
    return joblib.load("label_encoders.pkl")

st.set_page_config(layout="wide", page_title="Fraud Detection System", page_icon="💸")

model = load_model()

encoder = load_encoder()

def encode_data(col, input_data):
    for col in categorical_col:
        try:
            input_data[col] = encoder[col].transform(input_data[col])
        except ValueError:
            input_data[col] = -1

st.markdown(
    "<h1 style='color: ; text-align: center;'>Fraud Detection System</h1><br>",
    unsafe_allow_html=True
)

st.write("Enter the transaction details:")

merchant = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
job = st.text_input("Job")
age = st.number_input("Age", min_value=0.0)
gender = st.selectbox("Gender", ["Male","Female"])
city_pop = st.number_input("City Population", min_value=0.0)
lat = st.number_input("Transaction Latitude", min_value=0.0, format="%.4f")
long = st.number_input("Transaction Longitude", format="%.4f")
merch_lat = st.number_input("Merchant Latitude", min_value=0.0, format="%.4f")
merch_long = st.number_input("Merchant Longitude", format="%.4f")
hour = st.slider("Transaction Hour", 0, 23, 12)
day = st.slider("Transaction Day", 1, 31, 15)
month = st.slider("Transaction Month", 1, 12, 6)

categorical_col = ["merchant", "category", "gender", "job"]

if "fraud_prediction" not in st.session_state:
    st.session_state.fraud_prediction = False
if "legit_prediction" not in st.session_state:
    st.session_state.legit_prediction = False
if "manual_prediction" not in st.session_state:
    st.session_state.manual_prediction = False

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    if st.button("🔍 Check For Fraud", use_container_width=True):
        if merchant and category:
            input_data = pd.DataFrame([[merchant, category, amt, job, age, hour, day, month, gender, city_pop, lat, long, merch_lat, merch_long]],
                                columns=["merchant", "category", "amt", "job", "age", "hour", "day", "month", "gender", "city_pop", "lat", "long", "merch_lat", "merch_long"])
            
            encode_data(categorical_col, input_data)

            proba = model.predict_proba(input_data)[0, 1]
            result = ":green[Fraudulent Transaction]" if proba > 0.5 else ":red[Legitimate Transaction]"
            
            st.session_state.manual_prediction = True
            st.session_state.manual_result = result
            st.session_state.manual_proba = proba
        else:
            st.error("Please fill all required fields")

if st.session_state.manual_prediction:
    st.subheader(f"Prediction: {st.session_state.manual_result}")
    if st.session_state.manual_proba >= 0.5:
        st.subheader(f"Fraud Probability: :green[{st.session_state.manual_proba:.3f}]")
    else:
        st.subheader(f"Fraud Probability: :red[{st.session_state.manual_proba:.3f}]")

with col2:
    if st.button("⚠️ Generate Fraud Case", use_container_width=True):
        fraud_samples = load_fraud_samples()
        fraud_case = fraud_samples.sample(1)
        st.session_state["fraud_case"] = fraud_case
        
        fraud_case_encoded = fraud_case.copy()
        encode_data(categorical_col, fraud_case_encoded)
        pred = model.predict_proba(fraud_case_encoded)[0, 1]
        result = ":green[Fraudulent Transaction]" if pred > 0.5 else ":red[Legitimate Transaction]"
        
        st.session_state.fraud_prediction = True
        st.session_state.fraud_pred = pred
        st.session_state.fraud_result = result
        
if st.session_state.fraud_prediction:
    st.markdown('<h3><u>Fraud Input Data:</u></h3>', unsafe_allow_html=True)
    fraud_case_display = st.session_state.fraud_case.copy()
    fraud_case_display['amt'] = fraud_case_display['amt'].map('{:.2f}'.format)
    st.dataframe(fraud_case_display.reset_index(drop=True))
    st.subheader(f"Prediction: {st.session_state.fraud_result}")
    pred = st.session_state.fraud_pred
    if pred >= 0.5:
        st.subheader(f"Fraud Probability: :green[{pred:.3f}]")
    else:
        st.subheader(f"Fraud Probability: :red[{pred:.3f}]")

with col3:
    if st.button("✅ Generate Legit Case", use_container_width=True):
        legit_samples = load_legit_samples()
        legit_case = legit_samples.sample(1)
        st.session_state['legit_case'] = legit_case
        
        legit_case_encoded = legit_case.copy()
        encode_data(categorical_col, legit_case_encoded)
        pred = model.predict_proba(legit_case_encoded)[0, 1]
        result = ":green[Fraudulent Transaction]" if pred > 0.5 else ":red[Legitimate Transaction]"
        
        st.session_state.legit_prediction = True
        st.session_state.legit_pred = pred
        st.session_state.legit_result = result

if st.session_state.legit_prediction:
    st.markdown('<h3><u>Legit Input Data:</u></h3>', unsafe_allow_html=True)
    legit_case_display = st.session_state.legit_case.copy()
    legit_case_display['amt'] = legit_case_display['amt'].map('{:.2f}'.format)
    st.dataframe(legit_case_display.reset_index(drop=True))
    st.subheader(f"Prediction: {st.session_state.legit_result}")
    pred = st.session_state.legit_pred
    if pred >= 0.5:
        st.subheader(f"Fraud Probability: :green[{pred:.3f}]")
    else:
        st.subheader(f"Fraud Probability: :red[{pred:.3f}]")

