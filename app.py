import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
model = joblib.load("car_price_model.pkl")

# Training feature columns
feature_columns = joblib.load("feature_columns.pkl")

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
df = pd.read_csv("ford.csv")

# --------------------------------------------------
# Recreate the SAME preprocessing used in training
# --------------------------------------------------
X = df.drop(columns=["price"])

X_one_encode = pd.get_dummies(
    X,
    columns=["model", "transmission", "fuelType"],
    drop_first=True
)

numerical_cols = [
    "year",
    "mileage",
    "tax",
    "mpg",
    "engineSize"
]

# Recreate the scaler used during model training
scaler = StandardScaler()

X_one_encode[numerical_cols] = scaler.fit_transform(
    X_one_encode[numerical_cols]
)

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(
    page_title="Ford Price Predictor",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Ford Price Predictor")
st.write("Enter the vehicle specifications to estimate its price.")

st.divider()

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:

    car_model = st.selectbox(
        "Ford Model",
        sorted(df["model"].unique())
    )

    transmission = st.selectbox(
        "Transmission",
        sorted(df["transmission"].unique())
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        sorted(df["fuelType"].unique())
    )

    year = st.number_input(
        "Year",
        min_value=int(df["year"].min()),
        max_value=2026,
        value=2018,
        step=1
    )

with col2:

    mileage = st.number_input(
        "Mileage",
        min_value=0,
        max_value=int(df["mileage"].max()),
        value=20000,
        step=100
    )

    tax = st.number_input(
        "Tax",
        min_value=0,
        max_value=int(df["tax"].max()),
        value=145,
        step=1
    )

    mpg = st.number_input(
        "MPG",
        min_value=0.0,
        max_value=float(df["mpg"].max()),
        value=55.0,
        step=0.1
    )

    engine_size = st.number_input(
        "Engine Size (L)",
        min_value=0.1,
        max_value=float(df["engineSize"].max()),
        value=1.2,
        step=0.1
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("🔮 Predict Price", use_container_width=True):

    input_data = pd.DataFrame({
        "model": [car_model],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuel_type],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine_size]
    })

    # One-hot encode exactly like training
    input_encoded = pd.get_dummies(
        input_data,
        columns=["model", "transmission", "fuelType"],
        drop_first=True
    )

    # Match the exact 34 training features
    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale ONLY numerical columns
    input_encoded[numerical_cols] = scaler.transform(
        input_encoded[numerical_cols]
    )

    # Predict
    prediction = model.predict(input_encoded)[0]

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------
    st.success(f"### Estimated Price: £{prediction:,.2f}")

    st.subheader("Vehicle Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Model", car_model)
    c2.metric("Year", year)
    c3.metric("Mileage", f"{mileage:,}")
    c4.metric("Fuel", fuel_type)

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )