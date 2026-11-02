import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SolarIQ AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SIMPLE PAGE STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
    }

    [data-testid="stSidebar"] {
        background-color: #f1f6fb;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 18px;
        color: #667085;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 18px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        background-color: #eaf8f1;
        border: 1px solid #c9ead9;
        text-align: center;
    }

    .result-number {
        font-size: 38px;
        font-weight: 800;
        color: #148447;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATASET
# ============================================================

TARGET = "Estimated Annual PV Energy Production (kWh)"

possible_files = [
    "Solar Energy.csv",
    "Solar_Energy.csv",
    "solar_energy.csv",
    "Solar Energy (1).csv"
]

file_path = None

for file_name in possible_files:
    if os.path.exists(file_name):
        file_path = file_name
        break


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("☀️ SolarIQ AI")

    st.write(
        "AI-powered solar energy estimation platform."
    )

    st.divider()

    st.subheader("◉ AI MODEL")

    st.success("Linear Regression")

    st.subheader("📊 DATASET")

    if file_path:
        # temporary text, updated after loading
        st.info("Solar Energy Dataset")
    else:
        st.error("Dataset not found")

    st.subheader("⚡ SYSTEM")

    st.success("ONLINE")

    st.divider()

    st.caption("Solar Energy Intelligence Platform")


# ============================================================
# DATASET NOT FOUND
# ============================================================

if file_path is None:

    st.error(
        "Solar Energy.csv file nahi mili."
    )

    st.info(
        "app.py ke same folder mein "
        "'Solar Energy.csv' rakho."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_dataset(path):

    data = pd.read_csv(path)

    data.columns = data.columns.str.strip()

    return data


df = load_dataset(file_path)


# ============================================================
# CHECK TARGET
# ============================================================

if TARGET not in df.columns:

    st.error(
        "Target column nahi mili:"
    )

    st.code(TARGET)

    st.write("Available columns:")

    st.write(df.columns.tolist())

    st.stop()


# ============================================================
# TARGET CLEANING
# ============================================================

df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("$", "", regex=False)
    .str.strip()
)

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df.dropna(
    subset=[TARGET]
).copy()


# ============================================================
# FIND NUMERIC FEATURES
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


feature_columns = [
    column
    for column in numeric_columns
    if column != TARGET
]


# Remove columns with more than 80% missing values

usable_features = []

for column in feature_columns:

    missing_ratio = df[column].isna().mean()

    if missing_ratio < 0.80:
        usable_features.append(column)


feature_columns = usable_features


if len(feature_columns) == 0:

    st.error(
        "Model ke liye koi suitable numeric feature nahi mila."
    )

    st.stop()


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(data, features):

    X = data[features].copy()

    y = data[TARGET].copy()

    # Convert features to numeric

    for column in features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Imputation

    imputer = SimpleImputer(
        strategy="median"
    )

    X_processed = imputer.fit_transform(X)

    # Train/Test split

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=0.20,
        random_state=42
    )

    # Model

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    # Prediction

    y_pred = model.predict(
        X_test
    )

    # Metrics

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    return (
        model,
        imputer,
        mae,
        rmse,
        r2
    )


model, imputer, mae, rmse, r2 = train_model(
    df,
    feature_columns
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="main-title">☀️ SolarIQ AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Intelligent photovoltaic energy forecasting powered by Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "⚡ AI POWERED SOLAR ANALYTICS"
)


# ============================================================
# OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📊 Solar Intelligence Overview'
    '</div>',
    unsafe_allow_html=True
)


total_records = len(df)

average_energy = df[TARGET].mean()

total_features = len(df.columns)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "TOTAL PROJECTS",
        f"{total_records:,}"
    )

    st.caption(
        "Solar records analyzed"
    )


with col2:

    st.metric(
        "AVG ANNUAL ENERGY",
        f"{average_energy:,.0f} kWh"
    )

    st.caption(
        "Average estimated production"
    )


with col3:

    st.metric(
        "DATA FEATURES",
        total_features
    )

    st.caption(
        "Dataset columns"
    )


with col4:

    st.metric(
        "AI STATUS",
        "● ONLINE"
    )

    st.caption(
        "Linear Regression active"
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🤖 Model Performance'
    '</div>',
    unsafe_allow_html=True
)


m1, m2, m3 = st.columns(3)


with m1:

    st.metric(
        "R² SCORE",
        f"{r2:.4f}"
    )


with m2:

    st.metric(
        "MAE",
        f"{mae:,.2f} kWh"
    )


with m3:

    st.metric(
        "RMSE",
        f"{rmse:,.2f} kWh"
    )


# ============================================================
# PREDICTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔮 Solar Energy Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Solar system parameters enter karke annual "
    "PV energy production predict karo."
)


# ============================================================
# INPUTS
# ============================================================

input_values = {}

input_columns = st.columns(2)


for index, feature in enumerate(feature_columns):

    series = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    median_value = series.median()

    minimum = series.min()

    maximum = series.max()

    if pd.isna(median_value):
        median_value = 0.0

    if pd.isna(minimum):
        minimum = 0.0

    if pd.isna(maximum):
        maximum = median_value + 100

    # Same min/max ko avoid karna

    if minimum == maximum:

        minimum = minimum - 1

        maximum = maximum + 1

    with input_columns[index % 2]:

        input_values[feature] = st.number_input(
            feature,
            min_value=float(minimum),
            max_value=float(maximum),
            value=float(
                np.clip(
                    median_value,
                    minimum,
                    maximum
                )
            ),
            step=0.1
        )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.write("")

predict = st.button(
    "⚡ Predict Annual Solar Energy",
    type="primary",
    use_container_width=True
)


if predict:

    input_df = pd.DataFrame(
        [input_values]
    )

    # Same feature order as training

    input_df = input_df[
        feature_columns
    ]

    # Numeric conversion

    for column in feature_columns:

        input_df[column] = pd.to_numeric(
            input_df[column],
            errors="coerce"
        )

    # Apply same imputer

    input_processed = imputer.transform(
        input_df
    )

    # Prediction

    prediction = model.predict(
        input_processed
    )[0]

    # Negative energy possible nahi

    prediction = max(
        0,
        prediction
    )

    st.success(
        "Prediction generated successfully!"
    )

    st.markdown(
        '<div class="result-box">'
        '<p>ESTIMATED ANNUAL PV ENERGY PRODUCTION</p>'
        f'<div class="result-number">'
        f'{prediction:,.2f} kWh'
        f'</div>'
        '<p>Model: Linear Regression</p>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# FEATURE INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📌 Model Features'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    f"Linear Regression model mein "
    f"**{len(feature_columns)} numeric features** use ho rahe hain."
)

with st.expander("View Features Used by Model"):

    for number, feature in enumerate(
        feature_columns,
        start=1
    ):

        st.write(
            f"{number}. {feature}"
        )


# ============================================================
# DATASET PREVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📋 Dataset Preview'
    '</div>',
    unsafe_allow_html=True
)

with st.expander("View Dataset"):

    st.dataframe(
        df.head(100),
        use_container_width=True
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.expander("🔍 Dataset Information"):

    info1, info2 = st.columns(2)

    with info1:

        st.write(
            "**Rows:**",
            len(df)
        )

        st.write(
            "**Columns:**",
            len(df.columns)
        )

        st.write(
            "**Target:**",
            TARGET
        )

    with info2:

        st.write(
            "**Missing Values:**",
            int(df.isna().sum().sum())
        )

        st.write(
            "**Features Used:**",
            len(feature_columns)
        )

        st.write(
            "**Model:**",
            "Linear Regression"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "☀️ SolarIQ AI | Solar Energy Intelligence Platform | "
    "Machine Learning Powered"
)
