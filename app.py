import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


@st.cache_data
def load_fraud_samples():
    return pd.read_csv("fraud_samples.csv")


@st.cache_data
def load_legit_samples():
    return pd.read_csv("legit_samples.csv")


@st.cache_data
def load_dataset():
    return pd.read_csv("credit_card_transactions.csv")


@st.cache_data
def get_unique_values():
    dataset = load_dataset()
    return {
        "merchants": sorted(dataset["merchant"].unique().tolist()),
        "categories": sorted(dataset["category"].unique().tolist()),
        "jobs": sorted(dataset["job"].unique().tolist()),
        "states": sorted(dataset["state"].unique().tolist()),
    }


@st.cache_resource()
def load_model():
    return joblib.load("fraud_detection_model.pkl")


@st.cache_resource()
def load_encoder():
    return joblib.load("label_encoders.pkl")


st.set_page_config(layout="wide", page_title="Fraud Detection System", page_icon="💸")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
        
        body, p, span, div, label, input, button, select, textarea {
            font-family: 'Roboto', sans-serif !important;
            font-weight: 500 !important;
        }
        
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
        }
        
        [data-testid="stButton"] button {
            transition: all 0.25s ease !important;
        }
        
        [data-testid="stButton"] button:hover {
            transform: scale(1.02) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        }
        
        [data-testid="stButton"] button:active {
            transform: scale(0.99) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

model = load_model()

encoder = load_encoder()

unique_vals = get_unique_values()
merchants = unique_vals["merchants"]
categories = unique_vals["categories"]
jobs = unique_vals["jobs"]
states = unique_vals["states"]

if "prediction_type" not in st.session_state:
    st.session_state.prediction_type = None
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_proba" not in st.session_state:
    st.session_state.prediction_proba = None


def encode_data(col, input_data):
    for col in categorical_col:
        try:
            input_data[col] = encoder[col].transform(input_data[col])
        except ValueError:
            input_data[col] = -1


def decode_data(input_data):
    decoded_data = input_data.copy()
    for col in categorical_col:
        if col in decoded_data.columns:
            try:
                encoded_values = decoded_data[col].astype(int)
                decoded_data[col] = encoder[col].inverse_transform(encoded_values)
            except (ValueError, AttributeError):
                pass
    return decoded_data


def create_fraud_gauge(probability):
    color = "#ff4b4b" if probability >= 0.5 else "#21c354"
    bg_color = "rgba(200, 200, 200, 0.2)"

    fig = go.Figure()

    fig.add_trace(
        go.Pie(
            values=[100],
            hole=0.80,
            marker=dict(
                colors=[bg_color],
                line=dict(color="rgba(100, 100, 100, 0.4)", width=1.5),
            ),
            textinfo="none",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Pie(
            values=[probability * 100, 100 - (probability * 100)],
            hole=0.80,
            marker=dict(
                colors=[color, "rgba(0,0,0,0)"],
                line=dict(color="rgba(80, 80, 80, 0.5)", width=2.5),
            ),
            textinfo="none",
            hoverinfo="skip",
            showlegend=False,
            direction="clockwise",
            rotation=90,
            sort=False,
        )
    )

    fig.add_annotation(
        text=f"<b>{probability * 100:.1f}%</b>",
        x=0.5,
        y=0.52,
        font=dict(size=60, color=color, family="Roboto"),
        showarrow=False,
    )

    label_text = "FRAUD RISK" if probability >= 0.5 else "LEGITIMATE"
    fig.add_annotation(
        text=label_text,
        x=0.5,
        y=0.38,
        font=dict(size=14, color="rgba(150,150,150,0.9)", family="Roboto", weight=600),
        showarrow=False,
    )

    fig.update_layout(
        height=370,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    return fig


def create_feature_importance_chart(input_data):
    feature_names = [
        "Merchant",
        "Category",
        "Amount",
        "Job",
        "Age",
        "Hour",
        "Day",
        "Month",
        "Gender",
        "State",
        "City Pop",
        "Trans Lat",
        "Trans Long",
        "Merch Lat",
        "Merch Long",
    ]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        feature_importance_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=True)
            .tail(8)
        )

        custom_colorscale = [
            [0, "#8e99b3"],
            [0.5, "#5c6b8a"],
            [1, "#3d4d66"],
        ]

        fig = go.Figure(
            go.Bar(
                x=feature_importance_df["Importance"],
                y=feature_importance_df["Feature"],
                orientation="h",
                marker=dict(
                    color=feature_importance_df["Importance"],
                    colorscale=custom_colorscale,
                    showscale=False,
                ),
                text=[f"{x:.3f}" for x in feature_importance_df["Importance"]],
                textposition="auto",
            )
        )

        fig.update_layout(
            title="Top 8 Most Influential Features",
            xaxis_title="Importance Score",
            yaxis_title="",
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Roboto"},
        )

        return fig
    return None


st.markdown(
    "<h1 style='color: ; text-align: center;'>Fraud Detection System</h1><br>",
    unsafe_allow_html=True,
)

header_col1, header_spacer, header_col2 = st.columns([8, 8, 1])
with header_col1:
    st.write("Enter the transaction details:")
with header_col2:
    if st.button("Reset", type="secondary", help="Clear Results"):
        st.session_state.prediction_type = None
        st.session_state.prediction_data = None
        st.session_state.prediction_result = None
        st.session_state.prediction_proba = None
        st.session_state.merchant_select = ""
        st.session_state.category_select = ""
        st.session_state.job_select = ""
        st.session_state.state_select = ""
        st.session_state.amt_input = 0.0
        st.session_state.age_input = 0.0
        st.session_state.gender_select = "Male"
        st.session_state.city_pop_input = 0.0
        st.session_state.lat_input = 0.0
        st.session_state.long_input = 0.0
        st.session_state.merch_lat_input = 0.0
        st.session_state.merch_long_input = 0.0
        st.session_state.hour_slider = 12
        st.session_state.day_slider = 15
        st.session_state.month_slider = 6

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    merchant = st.selectbox(
        "Merchant Name",
        options=[""] + merchants,
        help="Business name (e.g., fraud_Koepp-Parker, fraud_Towne-Koepp)",
        key="merchant_select",
    )
with col2:
    category = st.selectbox(
        "Category",
        options=[""] + categories,
        help="Transaction category (e.g., gas_transport, shopping_net, grocery_pos)",
        key="category_select",
    )
with col3:
    job = st.selectbox(
        "Job",
        options=[""] + jobs,
        help="Occupation (e.g., Naval architect, Engineer)",
        key="job_select",
    )

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    state = st.selectbox(
        "State",
        options=[""] + states,
        help="Enter 2-letter state code (e.g., CA, NY, TX)",
        key="state_select",
    )
with col2:
    amt = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        format="%.2f",
        help="Purchase amount in USD",
        key="amt_input",
    )
with col3:
    age = st.number_input(
        "Age", min_value=0.0, help="Cardholder's age in years", key="age_input"
    )

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    gender = st.selectbox(
        "Gender", ["Male", "Female"], help="Cardholder's gender", key="gender_select"
    )
with col2:
    city_pop = st.number_input(
        "City Population",
        min_value=0.0,
        help="Population of the cardholder's city",
        key="city_pop_input",
    )
with col3:
    lat = st.number_input(
        "Transaction Latitude",
        min_value=0.0,
        format="%.4f",
        help="Latitude where transaction occurred",
        key="lat_input",
    )

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    long = st.number_input(
        "Transaction Longitude",
        format="%.4f",
        help="Longitude where transaction occurred",
        key="long_input",
    )
with col2:
    merch_lat = st.number_input(
        "Merchant Latitude",
        min_value=0.0,
        format="%.4f",
        help="Latitude of merchant's location",
        key="merch_lat_input",
    )
with col3:
    merch_long = st.number_input(
        "Merchant Longitude",
        format="%.4f",
        help="Longitude of merchant's location",
        key="merch_long_input",
    )

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    hour = st.slider(
        "Transaction Hour",
        0,
        23,
        12,
        help="Hour of day (0-23) when transaction occurred",
        key="hour_slider",
    )
with col2:
    day = st.slider(
        "Transaction Day",
        1,
        31,
        15,
        help="Day of month when transaction occurred",
        key="day_slider",
    )
with col3:
    month = st.slider(
        "Transaction Month",
        1,
        12,
        6,
        help="Month of year when transaction occurred",
        key="month_slider",
    )

categorical_col = ["merchant", "category", "gender", "job", "state"]

if "prediction_type" not in st.session_state:
    st.session_state.prediction_type = None
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_proba" not in st.session_state:
    st.session_state.prediction_proba = None

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    if st.button("Check For Fraud", use_container_width=True):
        if merchant and category and state:
            input_data = pd.DataFrame(
                [
                    [
                        merchant,
                        category,
                        amt,
                        job,
                        age,
                        hour,
                        day,
                        month,
                        gender,
                        state,
                        city_pop,
                        lat,
                        long,
                        merch_lat,
                        merch_long,
                    ]
                ],
                columns=[
                    "merchant",
                    "category",
                    "amt",
                    "job",
                    "age",
                    "hour",
                    "day",
                    "month",
                    "gender",
                    "state",
                    "city_pop",
                    "lat",
                    "long",
                    "merch_lat",
                    "merch_long",
                ],
            )

            encode_data(categorical_col, input_data)

            proba = model.predict_proba(input_data)[0, 1]
            result = (
                ":green[Fraudulent Transaction]"
                if proba > 0.5
                else ":red[Legitimate Transaction]"
            )

            st.session_state.prediction_type = "manual"
            st.session_state.prediction_data = input_data
            st.session_state.prediction_result = result
            st.session_state.prediction_proba = proba
        else:
            st.toast(
                "Please fill all required fields: Merchant Name, Category, and State"
            )

with col2:
    if st.button("Generate Fraud Case", use_container_width=True):
        fraud_samples = load_fraud_samples()
        fraud_case = fraud_samples.sample(1)

        fraud_case_encoded = fraud_case.copy()
        encode_data(categorical_col, fraud_case_encoded)
        pred = model.predict_proba(fraud_case_encoded)[0, 1]
        result = (
            ":green[Fraudulent Transaction]"
            if pred > 0.5
            else ":red[Legitimate Transaction]"
        )

        st.session_state.prediction_type = "fraud"
        st.session_state.prediction_data = fraud_case
        st.session_state.prediction_result = result
        st.session_state.prediction_proba = pred

with col3:
    if st.button("Generate Legit Case", use_container_width=True):
        legit_samples = load_legit_samples()
        legit_case = legit_samples.sample(1)

        legit_case_encoded = legit_case.copy()
        encode_data(categorical_col, legit_case_encoded)
        pred = model.predict_proba(legit_case_encoded)[0, 1]
        result = (
            ":green[Fraudulent Transaction]"
            if pred > 0.5
            else ":red[Legitimate Transaction]"
        )

        st.session_state.prediction_type = "legit"
        st.session_state.prediction_data = legit_case
        st.session_state.prediction_result = result
        st.session_state.prediction_proba = pred

if st.session_state.prediction_type is not None:
    st.markdown(
        "<hr style='margin-top: 0.75rem; margin-bottom: 1.25rem;'>",
        unsafe_allow_html=True,
    )

    proba_percent = st.session_state.prediction_proba * 100

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.plotly_chart(
            create_fraud_gauge(st.session_state.prediction_proba),
            use_container_width=True,
        )

        if st.session_state.prediction_proba >= 0.5:
            st.markdown(
                f"""
                <div style='background-color: rgba(255, 75, 75, 0.1); padding: 15px; border-radius: 5px; border-left: 5px solid #ff4b4b; text-align: center;'>
                    <span style='color: #ff4b4b; font-weight: 600; font-size: 1.1em;'>Fraudulent Transaction</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style='background-color: rgba(33, 195, 84, 0.1); padding: 15px; border-radius: 5px; border-left: 5px solid #21c354; text-align: center;'>
                    <span style='color: #21c354; font-weight: 600; font-size: 1.1em;'>Legitimate Transaction</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        importance_chart = create_feature_importance_chart(None)
        if importance_chart:
            st.plotly_chart(importance_chart, use_container_width=True)

    if st.session_state.prediction_data is not None:
        st.markdown(
            "<hr style='margin-top: 1rem; margin-bottom: 0.5rem;'>",
            unsafe_allow_html=True,
        )

        if st.session_state.prediction_type == "fraud":
            case_type = "Fraud Sample"
        elif st.session_state.prediction_type == "legit":
            case_type = "Legit Sample"
        else:
            case_type = "Manual Input"

        st.subheader(f"{case_type} Data")
        case_display = st.session_state.prediction_data.copy()

        case_display = decode_data(case_display)

        case_display["amt"] = case_display["amt"].map("{:.2f}".format)

        column_order = [
            "merchant",
            "category",
            "job",
            "state",
            "amt",
            "age",
            "gender",
            "city_pop",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
            "hour",
            "day",
            "month",
        ]
        case_display = case_display[column_order]

        st.dataframe(case_display.reset_index(drop=True), use_container_width=True)
