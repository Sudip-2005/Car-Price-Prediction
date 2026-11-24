"""
🚗 Used Car Price Predictor — Streamlit app
Author: GTA VI

Enter a car's details and get an estimated resale price, powered by a
Random Forest model trained on 15,000+ CarDekho listings.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "Data/cardekho_imputated.csv"
CATEGORICAL = ["model", "seller_type", "fuel_type", "transmission_type"]
NUMERIC = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
TARGET = "selling_price"

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Custom Styling ----------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f766e 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.15);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 0.98rem;
        color: #cbd5e1;
        margin: 0 0 1rem 0;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .badge {
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: #f8fafc;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .price-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdfa 100%);
        border: 1px solid #99f6e4;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1rem;
    }
    .price-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #0f766e;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .price-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }
    .price-sub {
        font-size: 0.95rem;
        color: #334155;
        font-weight: 500;
    }
    @media (prefers-color-scheme: dark) {
        .price-card {
            background: linear-gradient(135deg, #064e3b 0%, #115e59 100%);
            border-color: #14b8a6;
        }
        .price-label { color: #99f6e4; }
        .price-value { color: #ffffff; }
        .price-sub { color: #ccfbf1; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Helpers ----------
def format_inr(amount: float) -> str:
    """Format a number using the Indian numbering system (e.g., ₹ 6,45,000)."""
    n = max(0, int(round(amount)))
    s = str(n)
    if len(s) <= 3:
        return f"₹ {s}"
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.append(rest)
    parts.reverse()
    return f"₹ {','.join(parts)},{last3}"


def format_lakhs(amount: float) -> str:
    if amount >= 1e7:
        return f"₹ {amount / 1e7:.2f} Cr"
    return f"₹ {amount / 1e5:.2f} Lakh"


def clamp(val, low, high):
    return max(low, min(high, val))


@st.cache_data
def load_data():
    return pd.read_csv(DATA, index_col=[0])


@st.cache_resource
def train_model(df):
    """Train the pipeline once and cache it for the life of the container."""
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    pipeline = Pipeline(
        [
            ("prep", preprocessor),
            ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def predict_with_range(pipeline, row_df):
    """Return point estimate and 25th–75th percentile range across RF trees."""
    X_trans = pipeline.named_steps["prep"].transform(row_df)
    rf = pipeline.named_steps["rf"]
    tree_preds = np.array([est.predict(X_trans)[0] for est in rf.estimators_])
    point = float(tree_preds.mean())
    low, high = np.percentile(tree_preds, [25, 75])
    return point, float(low), float(high)


# ---------- Load Data & Model ----------
df = load_data()
with st.spinner("Warming up the Random Forest model (first run only)…"):
    model = train_model(df)

# Global slider bounds
AGE_MIN, AGE_MAX = int(df["vehicle_age"].min()), int(df["vehicle_age"].max())
KM_MIN, KM_MAX = 0, int(df["km_driven"].quantile(0.99))
MIL_MIN, MIL_MAX = float(round(df["mileage"].min(), 1)), float(round(df["mileage"].max(), 1))
ENG_MIN, ENG_MAX = int(df["engine"].min()), int(df["engine"].quantile(0.99))
PWR_MIN, PWR_MAX = float(round(df["max_power"].min(), 1)), float(round(df["max_power"].quantile(0.99), 1))

MODELS = sorted(df["model"].unique())
FUELS = sorted(df["fuel_type"].unique())
TRANSMISSIONS = sorted(df["transmission_type"].unique())
SELLERS = sorted(df["seller_type"].unique())
SEATS_LIST = sorted(int(s) for s in df["seats"].unique())


def apply_model_defaults():
    """Auto-populate realistic defaults whenever the selected car model changes."""
    sub = df[df["model"] == st.session_state["model_name"]]
    if sub.empty:
        return
    st.session_state["fuel_type"] = sub["fuel_type"].mode().iloc[0]
    st.session_state["transmission_type"] = sub["transmission_type"].mode().iloc[0]
    st.session_state["seller_type"] = sub["seller_type"].mode().iloc[0]
    st.session_state["seats"] = int(sub["seats"].mode().iloc[0])
    st.session_state["vehicle_age"] = int(clamp(round(sub["vehicle_age"].median()), AGE_MIN, AGE_MAX))
    st.session_state["km_driven"] = int(clamp(round(sub["km_driven"].median(), -3), KM_MIN, KM_MAX))
    st.session_state["mileage"] = float(clamp(round(sub["mileage"].median(), 1), MIL_MIN, MIL_MAX))
    st.session_state["engine"] = int(clamp(round(sub["engine"].median() / 50) * 50, ENG_MIN, ENG_MAX))
    st.session_state["max_power"] = float(clamp(round(sub["max_power"].median(), 1), PWR_MIN, PWR_MAX))


# Initialize session state on first load
if "model_name" not in st.session_state:
    st.session_state["model_name"] = MODELS[0]
    apply_model_defaults()


# ---------- Hero Banner ----------
st.markdown(
    f"""
    <div class="hero-box">
        <div class="hero-title">🚗 Used Car Price Predictor</div>
        <div class="hero-sub">
            Instant resale valuation for used cars in India, powered by machine learning.
            Built by <strong>GTA VI</strong>.
        </div>
        <div class="badge-row">
            <span class="badge">📊 {len(df):,} CarDekho Listings</span>
            <span class="badge">🌲 Random Forest (~0.93 R²)</span>
            <span class="badge">⚡ Live Spec Auto-Fill</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Main Layout ----------
left_col, right_col = st.columns([1.05, 1.35], gap="large")

with left_col:
    with st.container(border=True):
        st.subheader("1. Car Details")
        st.selectbox(
            "Car model",
            MODELS,
            key="model_name",
            on_change=apply_model_defaults,
            help="Changing the model automatically loads its typical specs.",
        )

        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("Fuel type", FUELS, key="fuel_type")
            st.selectbox("Seller type", SELLERS, key="seller_type")
        with c2:
            st.selectbox("Transmission", TRANSMISSIONS, key="transmission_type")
            st.selectbox("Seats", SEATS_LIST, key="seats")

    with st.container(border=True):
        st.subheader("2. Usage & Specifications")
        st.slider("Vehicle age (years)", AGE_MIN, AGE_MAX, key="vehicle_age")
        st.slider("Kilometres driven", KM_MIN, KM_MAX, step=1000, key="km_driven")

        s1, s2 = st.columns(2)
        with s1:
            st.slider("Mileage (kmpl)", MIL_MIN, MIL_MAX, step=0.5, key="mileage")
            st.slider("Engine (cc)", ENG_MIN, ENG_MAX, step=50, key="engine")
        with s2:
            st.slider("Max power (bhp)", PWR_MIN, PWR_MAX, step=1.0, key="max_power")

        if st.button("↺ Reset to model defaults", use_container_width=True):
            apply_model_defaults()
            st.rerun()


# ---------- Build Prediction Row ----------
input_row = pd.DataFrame(
    [
        {
            "model": st.session_state["model_name"],
            "seller_type": st.session_state["seller_type"],
            "fuel_type": st.session_state["fuel_type"],
            "transmission_type": st.session_state["transmission_type"],
            "vehicle_age": st.session_state["vehicle_age"],
            "km_driven": st.session_state["km_driven"],
            "mileage": st.session_state["mileage"],
            "engine": st.session_state["engine"],
            "max_power": st.session_state["max_power"],
            "seats": st.session_state["seats"],
        }
    ]
)

price, low_est, high_est = predict_with_range(model, input_row)
model_df = df[df["model"] == st.session_state["model_name"]]
model_median = float(model_df[TARGET].median())
diff_pct = ((price - model_median) / model_median * 100) if model_median > 0 else 0.0


# ---------- Right Column: Valuation & Insights ----------
with right_col:
    st.markdown(
        f"""
        <div class="price-card">
            <div class="price-label">Estimated Resale Value · {st.session_state['model_name']}</div>
            <div class="price-value">{format_inr(price)}</div>
            <div class="price-sub">
                {format_lakhs(price)} &nbsp;·&nbsp; Fair market range:
                <strong>{format_lakhs(low_est)} – {format_lakhs(high_est)}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Model Median Price",
        format_lakhs(model_median),
        f"{diff_pct:+.1f}% vs median",
    )
    m2.metric("Listings in Dataset", f"{len(model_df):,}")
    m3.metric("Typical Age", f"{int(model_df['vehicle_age'].median())} yrs")

    tab_forecast, tab_market, tab_comps = st.tabs(
        ["📉 Depreciation Forecast", "📊 Market Position", "🔎 Comparable Listings"]
    )

    with tab_forecast:
        st.caption(
            "Projected value if you keep this car for the next 5 years "
            "(assuming +10,000 km driven per year)."
        )
        future_rows = []
        for yr in range(0, 6):
            r = input_row.copy()
            r["vehicle_age"] = st.session_state["vehicle_age"] + yr
            r["km_driven"] = st.session_state["km_driven"] + yr * 10_000
            pred_yr = float(model.predict(r)[0])
            future_rows.append(
                {
                    "Year": "Now" if yr == 0 else f"+{yr} yr",
                    "Order": yr,
                    "Price (Lakhs)": round(pred_yr / 1e5, 2),
                    "Formatted": format_inr(pred_yr),
                }
            )
        forecast_df = pd.DataFrame(future_rows)

        chart = (
            alt.Chart(forecast_df)
            .mark_area(
                line={"color": "#0f766e", "strokeWidth": 3},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="rgba(15, 118, 110, 0.35)", offset=0),
                        alt.GradientStop(color="rgba(15, 118, 110, 0.02)", offset=1),
                    ],
                    x1=1,
                    x2=1,
                    y1=1,
                    y2=0,
                ),
                point={"filled": True, "fill": "#0f766e", "size": 65},
            )
            .encode(
                x=alt.X("Year:N", sort=list(forecast_df["Year"]), title="Timeline"),
                y=alt.Y("Price (Lakhs):Q", title="Estimated Price (₹ Lakhs)"),
                tooltip=["Year", "Formatted"],
            )
            .properties(height=270)
        )
        st.altair_chart(chart, use_container_width=True)

    with tab_market:
        st.caption(
            f"How your {st.session_state['model_name']} compares against "
            f"all {len(model_df):,} listings of the same model."
        )
        plot_df = model_df[["vehicle_age", TARGET, "km_driven", "fuel_type"]].copy()
        plot_df["Price (Lakhs)"] = (plot_df[TARGET] / 1e5).round(2)

        scatter = (
            alt.Chart(plot_df)
            .mark_circle(size=55, opacity=0.45, color="#64748b")
            .encode(
                x=alt.X("vehicle_age:Q", title="Vehicle Age (years)"),
                y=alt.Y("Price (Lakhs):Q", title="Selling Price (₹ Lakhs)"),
                tooltip=["vehicle_age", "Price (Lakhs)", "km_driven", "fuel_type"],
            )
        )
        user_point = (
            alt.Chart(
                pd.DataFrame(
                    [
                        {
                            "vehicle_age": st.session_state["vehicle_age"],
                            "Price (Lakhs)": round(price / 1e5, 2),
                            "Label": f"Your Estimate: {format_lakhs(price)}",
                        }
                    ]
                )
            )
            .mark_point(size=220, filled=True, color="#ef4444", shape="diamond")
            .encode(
                x="vehicle_age:Q",
                y="Price (Lakhs):Q",
                tooltip=["Label"],
            )
        )
        st.altair_chart((scatter + user_point).properties(height=270), use_container_width=True)

    with tab_comps:
        st.caption("Closest real listings in the dataset for this model.")
        comps = model_df.copy()
        comps["age_diff"] = (comps["vehicle_age"] - st.session_state["vehicle_age"]).abs()
        comps["km_diff"] = (comps["km_driven"] - st.session_state["km_driven"]).abs()
        comps = comps.sort_values(["age_diff", "km_diff"]).head(6)

        display_comps = pd.DataFrame(
            {
                "Age (yrs)": comps["vehicle_age"].astype(int),
                "KM Driven": comps["km_driven"].map(lambda x: f"{int(x):,}"),
                "Fuel": comps["fuel_type"],
                "Transmission": comps["transmission_type"],
                "Seller": comps["seller_type"],
                "Listed Price": comps[TARGET].map(format_inr),
            }
        )
        st.dataframe(display_comps, hide_index=True, use_container_width=True)

    st.caption(
        "Estimate only — actual resale prices vary with vehicle condition, city, "
        "ownership history, and demand. Model MAE is ≈ ₹1,00,000."
    )

st.divider()
st.caption(
    "Data: CarDekho listings · Model: Random Forest Regressor · "
    "[Source code](https://github.com/Sudip-2005/Car-Price-Prediction)"
)