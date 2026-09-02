import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Load trained pipeline + reference dataframe
# ---------------------------------------------------------
pipe = pickle.load(open("pipe.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))

st.title("💻 Laptop Price Predictor")
st.write("Fill in the specs below to estimate the laptop's price.")

# ---------------------------------------------------------
# Inputs (built from the dataset's own columns/categories)
# ---------------------------------------------------------
company = st.selectbox("Brand", sorted(df["Company"].unique()))
type_name = st.selectbox("Type", sorted(df["TypeName"].unique()))
ram = st.selectbox("RAM (GB)", sorted(df["Ram"].unique()))
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
ips = st.selectbox("IPS Display", ["No", "Yes"])
ppi = st.number_input("Screen PPI (pixel density)", min_value=90.0, max_value=400.0, value=141.0, step=1.0)
cpu = st.selectbox("CPU", sorted(df["Cpu"].unique()))
gpu = st.selectbox("GPU", sorted(df["Gpu"].unique()))
os = st.selectbox("Operating System", sorted(df["OpSys"].unique()))

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
# Streamlit re-runs this whole script on every interaction, but the code
# inside this block only executes on the run where the button was clicked.
if st.button("Predict Price"):

    # The dropdowns show "Yes"/"No" for readability, but the model was
    # trained on raw 0/1 integers for these two columns — convert back.
    touchscreen_val = 1 if touchscreen == "Yes" else 0
    ips_val = 1 if ips == "Yes" else 0

    # Build a single-row DataFrame from all the widget values.
    # - pipe.predict() needs a 2D input (rows x columns), so the dict is
    #   wrapped in a list "[{...}]" to make exactly one row.
    # - Column names here must exactly match the columns the
    #   ColumnTransformer was fit on in train_model.py.
    query = pd.DataFrame([{
        "Company": company,
        "TypeName": type_name,
        "Cpu": cpu,
        "Ram": ram,
        "Gpu": gpu,
        "OpSys": os,
        "Weight": weight,
        "Touchscreen": touchscreen_val,
        "IPS": ips_val,
        "ppi": ppi,
    }])

    # pipe is the full saved pipeline: it first one-hot encodes the
    # categorical columns (Company, TypeName, Cpu, Gpu, OpSys) via the
    # ColumnTransformer, passes the numeric columns through untouched,
    # then feeds the result into the Gradient Boosting model.
    # .predict() always returns an array, even for one row, so [0]
    # pulls out the single predicted price as a plain number.
    predicted_price = pipe.predict(query)[0]

    # Display the result. ":,.0f" adds comma separators and rounds to
    # a whole number, e.g. 77814.93 -> 77,815.
    st.success(f"Estimated Price: ₹{predicted_price:,.0f}")