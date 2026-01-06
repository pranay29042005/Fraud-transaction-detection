import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("AIML Dataset.csv")
    df = df.drop(columns=["nameOrig", "nameDest"])
    df["type"] = LabelEncoder().fit_transform(df["type"])
    df = df.fillna(0)
    return df

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(df):
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, _, y_train, _ = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model, scaler, X.columns

# ---------------- LOAD & TRAIN ----------------
df = load_data()

with st.spinner("🔄 Training fraud detection model..."):
    model, scaler, columns = train_model(df)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go To", ["Dashboard", "Predict"])

# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.markdown("""
    <style>
    .hero {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 35px;
        border-radius: 20px;
        color: white;
        text-align: center;
    }
    .card {
        background-color: #1f2933;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <h1>💳 Fraud Transaction Detection System</h1>
        <p>AI-powered real-time fraud detection dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='card'><h2>{len(df):,}</h2>Total Transactions</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><h2>{df['isFraud'].sum():,}</h2>Fraud Transactions</div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><h2>{len(df)-df['isFraud'].sum():,}</h2>Legitimate Transactions</div>", unsafe_allow_html=True)

    st.write("### 📈 Fraud vs Legitimate Transactions")

    fig, ax = plt.subplots()
    ax.bar(
        ["Legitimate", "Fraud"],
        [len(df)-df['isFraud'].sum(), df['isFraud'].sum()]
    )
    st.pyplot(fig)

# ---------------- PREDICTION PAGE ----------------
else:
    st.title("🔍 Predict Transaction Risk")

    user_data = []
    for col in columns:
        user_data.append(st.number_input(col, value=0.0))

    col1, col2 = st.columns(2)
    predict_btn = col1.button("🔍 Predict")
    reset_btn = col2.button("🔄 Reset")

    if predict_btn:
        arr = np.array(user_data).reshape(1, -1)
        arr = scaler.transform(arr)

        pred = model.predict(arr)[0]
        prob = model.predict_proba(arr)[0][1]

        st.write("### 🎯 Risk Score")
        st.progress(int(prob * 100))
        st.write(f"**Fraud Probability:** {prob*100:.2f}%")

        if pred == 1:
            st.error("🚨 Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")

    if reset_btn:
        st.experimental_rerun()
