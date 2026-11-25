"""
🚗 Used Car Price Predictor — Interactive Streamlit App
Author: GTA VI

Enter a car's details and get an instant resale valuation, depreciation forecast,
price driver breakdown, market positioning, and side-by-side car comparisons,
powered by a Random Forest model trained on 15,000+ CarDekho listings.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
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
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1240px;
    }
    .hero-box {
        background: linear-gradient(135deg, #0b132b 0%, #1c2541 55%, #0f766e 100%);
        padding: 1.75rem 2rem;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 1.25rem;
        box-shadow: 0 12px 30px rgba(11, 19, 43, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-title {
        font-size: 2.05rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.025em;
    }
    .hero-sub {
        font-size: 0.98rem;
        color: #cbd5e1;
        margin: 0 0 1rem 0;
        line-height: 1.45;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .badge {
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #f8fafc;
        padding: 0.28rem 0.78rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        backdrop-filter: blur(6px);
    }
    .price-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdfa 100%);
        border: 1px solid #99f6e4;
        border-radius: 18px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 6px 20px rgba(15, 118, 110, 0.08);
    }
    .price-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #0f766e;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .price-value {
        font-size: 2.65rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }
    .price-sub {
        font-size: 0.95rem;
        color: #334155;
        font-weight: 500;
    }
    .range-track {
        position: relative;
        height: 10px;
        background: rgba(15, 118, 110, 0.16);
        border-radius: 999px;
        margin-top: 0.9rem;
        overflow: hidden;
    }
    .range-fill {
        position: absolute;
        top: 0;
        height: 100%;
        background: linear-gradient(90deg, #14b8a6, #0f766e);
        border-radius: 999px;
    }
    .range-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        color: #475569;
        margin-top: 0.35rem;
        font-weight: 600;
    }
    .section-hint {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: -0.4rem;
        margin-bottom: 0.6rem;
    }
    @media (prefers-color-scheme: dark) {
        .price-card {
            background: linear-gradient(135deg, #064e3b 0%, #115e59 100%);
            border-color: #14b8a6;
        }
        .price-label { color: #99f6e4; }
        .price-value { color: #ffffff; }
        .price-sub { color: #ccfbf1; }
        .range-track { background: rgba(255, 255, 255, 0.18); }
        .range-labels { color: #cbd5e1; }
        .section-hint { color: #94a3b8; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Formatting & Utility Helpers ----------
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
    """Format an INR amount into Lakhs or Crores."""
    if amount >= 1e7:
        return f"₹ {amount / 1e7:.2f} Cr"
    return f"₹ {amount / 1e5:.2f} Lakh"


def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, index_col=[0])
    df["brand"] = df["brand"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train the Random Forest pipeline once and cache it with OOB metrics."""
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    rf = RandomForestRegressor(
        n_estimators=180,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline([("prep", preprocessor), ("rf", rf)])
    pipeline.fit(X, y)

    oob_preds = rf.oob_prediction_
    oob_r2 = float(r2_score(y, oob_preds))
    oob_mae = float(mean_absolute_error(y, oob_preds))
    return pipeline, oob_r2, oob_mae


def predict_with_range(pipeline: Pipeline, row_df: pd.DataFrame) -> tuple[float, float, float]:
    """Return point estimate and 25th–75th percentile range across RF trees."""
    X_trans = pipeline.named_steps["prep"].transform(row_df)
    rf = pipeline.named_steps["rf"]
    tree_preds = np.array([est.predict(X_trans)[0] for est in rf.estimators_])
    point = float(tree_preds.mean())
    low, high = np.percentile(tree_preds, [25, 75])
    return point, float(low), float(high)


def build_input_row(
    model_name: str,
    seller_type: str,
    fuel_type: str,
    transmission_type: str,
    vehicle_age: int,
    km_driven: int,
    mileage: float,
    engine: int,
    max_power: float,
    seats: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "seller_type": seller_type,
                "fuel_type": fuel_type,
                "transmission_type": transmission_type,
                "vehicle_age": vehicle_age,
                "km_driven": km_driven,
                "mileage": mileage,
                "engine": engine,
                "max_power": max_power,
                "seats": seats,
            }
        ]
    )


def compute_sensitivity(
    pipeline: Pipeline,
    base_row: pd.DataFrame,
    base_price: float,
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    """Measure how changing individual attributes shifts the predicted price."""
    scenarios = []
    age = int(base_row["vehicle_age"].iloc[0])
    km = int(base_row["km_driven"].iloc[0])
    fuel = base_row["fuel_type"].iloc[0]
    trans = base_row["transmission_type"].iloc[0]
    seller = base_row["seller_type"].iloc[0]

    candidates = [
        ("2 years newer", {"vehicle_age": max(0, age - 2)}),
        ("2 years older", {"vehicle_age": min(AGE_MAX, age + 2)}),
        ("20,000 km less driven", {"km_driven": max(0, km - 20_000)}),
        ("20,000 km more driven", {"km_driven": min(KM_MAX, km + 20_000)}),
    ]

    alt_trans = "Automatic" if trans == "Manual" else "Manual"
    if alt_trans in model_df["transmission_type"].values:
        candidates.append((f"Switch to {alt_trans}", {"transmission_type": alt_trans}))

    for alt_fuel in ["Diesel", "Petrol", "CNG"]:
        if alt_fuel != fuel and alt_fuel in model_df["fuel_type"].values:
            candidates.append((f"Switch to {alt_fuel}", {"fuel_type": alt_fuel}))
            break

    alt_seller = "Dealer" if seller == "Individual" else "Individual"
    candidates.append((f"Sell via {alt_seller}", {"seller_type": alt_seller}))

    for label, overrides in candidates:
        test_row = base_row.copy()
        for col, val in overrides.items():
            test_row[col] = val
        new_price = float(pipeline.predict(test_row)[0])
        delta = new_price - base_price
        scenarios.append(
            {
                "Factor": label,
                "Delta": delta,
                "New Price": new_price,
                "Direction": "Increases Value" if delta >= 0 else "Decreases Value",
                "Delta Label": f"{'+' if delta >= 0 else '-'}{format_lakhs(abs(delta))}",
            }
        )

    sens_df = pd.DataFrame(scenarios)
    sens_df["AbsDelta"] = sens_df["Delta"].abs()
    return sens_df.sort_values("AbsDelta", ascending=True)


# ---------- Load Data & Model ----------
df = load_data()
with st.spinner("Warming up the Random Forest model (first run only)…"):
    model, OOB_R2, OOB_MAE = train_model(df)

AGE_MIN, AGE_MAX = int(df["vehicle_age"].min()), int(df["vehicle_age"].max())
KM_MIN, KM_MAX = 0, int(df["km_driven"].quantile(0.99))
MIL_MIN, MIL_MAX = float(round(df["mileage"].min(), 1)), float(round(df["mileage"].max(), 1))
ENG_MIN, ENG_MAX = int(df["engine"].min()), int(df["engine"].quantile(0.99))
PWR_MIN, PWR_MAX = float(round(df["max_power"].min(), 1)), float(round(df["max_power"].quantile(0.99), 1))

BRANDS = ["All Brands"] + sorted(df["brand"].unique())
MODELS_ALL = sorted(df["model"].unique())
FUELS = sorted(df["fuel_type"].unique())
TRANSMISSIONS = sorted(df["transmission_type"].unique())
SELLERS = sorted(df["seller_type"].unique())
SEATS_LIST = sorted(int(s) for s in df["seats"].unique())


def get_models_for_brand(brand: str) -> list[str]:
    if brand == "All Brands":
        return MODELS_ALL
    return sorted(df[df["brand"] == brand]["model"].unique())


def apply_model_defaults(target_model: str | None = None):
    """Auto-populate realistic defaults for the chosen car model."""
    m = target_model or st.session_state["model_name"]
    sub = df[df["model"] == m]
    if sub.empty:
        return
    st.session_state["model_name"] = m
    st.session_state["fuel_type"] = sub["fuel_type"].mode().iloc[0]
    st.session_state["transmission_type"] = sub["transmission_type"].mode().iloc[0]
    st.session_state["seller_type"] = sub["seller_type"].mode().iloc[0]
    st.session_state["seats"] = int(sub["seats"].mode().iloc[0])
    st.session_state["vehicle_age"] = int(clamp(round(sub["vehicle_age"].median()), AGE_MIN, AGE_MAX))
    st.session_state["km_driven"] = int(clamp(round(sub["km_driven"].median(), -3), KM_MIN, KM_MAX))
    st.session_state["mileage"] = float(clamp(round(sub["mileage"].median(), 1), MIL_MIN, MIL_MAX))
    st.session_state["engine"] = int(clamp(round(sub["engine"].median() / 50) * 50, ENG_MIN, ENG_MAX))
    st.session_state["max_power"] = float(clamp(round(sub["max_power"].median(), 1), PWR_MIN, PWR_MAX))


def on_brand_change():
    brand = st.session_state["brand_filter"]
    models = get_models_for_brand(brand)
    if models:
        apply_model_defaults(models[0])


def apply_preset(brand: str, model_name: str, overrides: dict | None = None):
    st.session_state["brand_filter"] = brand
    apply_model_defaults(model_name)
    if overrides:
        for k, v in overrides.items():
            st.session_state[k] = v


# Initialize session state on first load
if "brand_filter" not in st.session_state:
    st.session_state["brand_filter"] = "Maruti"
if "model_name" not in st.session_state:
    default_model = "Swift" if "Swift" in MODELS_ALL else MODELS_ALL[0]
    apply_model_defaults(default_model)
if "saved_cars" not in st.session_state:
    st.session_state["saved_cars"] = []


# ---------- Hero Banner ----------
st.markdown(
    f"""
    <div class="hero-box">
        <div class="hero-title">🚗 Used Car Price Predictor</div>
        <div class="hero-sub">
            Instant resale valuation, depreciation modeling, and market intelligence for used cars in India.
            Built by <strong>GTA VI</strong>.
        </div>
        <div class="badge-row">
            <span class="badge">📊 {len(df):,} CarDekho Listings</span>
            <span class="badge">🌲 Random Forest · OOB R² {OOB_R2:.2f}</span>
            <span class="badge">🎯 Avg Error ≈ {format_lakhs(OOB_MAE)}</span>
            <span class="badge">⚡ Live Spec Auto-Fill</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Top-Level Workspace Tabs ----------
saved_count = len(st.session_state["saved_cars"])
compare_label = f"⚖️ Compare Cars ({saved_count})" if saved_count else "⚖️ Compare Cars"

page_predict, page_compare, page_explore = st.tabs(
    ["🚗 Price Predictor", compare_label, "📈 Market Explorer"]
)


# ==============================================================================
# TAB 1: PRICE PREDICTOR
# ==============================================================================
with page_predict:
    # Quick Presets Bar
    p_cols = st.columns([1.2, 1, 1, 1, 1])
    p_cols[0].markdown("**Quick presets:**")
    if p_cols[1].button("🚙 Maruti Swift", use_container_width=True):
        apply_preset("Maruti", "Swift", {"vehicle_age": 4, "km_driven": 35000})
        st.rerun()
    if p_cols[2].button("🏔️ Hyundai Creta", use_container_width=True):
        apply_preset("Hyundai", "Creta", {"vehicle_age": 3, "km_driven": 30000})
        st.rerun()
    if p_cols[3].button("🚘 Honda City", use_container_width=True):
        apply_preset("Honda", "City", {"vehicle_age": 5, "km_driven": 45000})
        st.rerun()
    if p_cols[4].button("👑 Toyota Fortuner", use_container_width=True):
        apply_preset("Toyota", "Fortuner", {"vehicle_age": 4, "km_driven": 50000})
        st.rerun()

    left_col, right_col = st.columns([1.05, 1.35], gap="large")

    with left_col:
        with st.container(border=True):
            st.subheader("1. Brand & Model")
            b_col, m_col = st.columns(2)
            with b_col:
                st.selectbox(
                    "Brand",
                    BRANDS,
                    key="brand_filter",
                    on_change=on_brand_change,
                )
            available_models = get_models_for_brand(st.session_state["brand_filter"])
            if st.session_state["model_name"] not in available_models:
                apply_model_defaults(available_models[0])
            with m_col:
                st.selectbox(
                    "Car model",
                    available_models,
                    key="model_name",
                    on_change=apply_model_defaults,
                    help="Selecting a model automatically fills its typical market specs.",
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
            st.slider(
                "Kilometres driven",
                KM_MIN,
                KM_MAX,
                step=1000,
                format="%d km",
                key="km_driven",
            )

            s1, s2 = st.columns(2)
            with s1:
                st.slider("Mileage (kmpl)", MIL_MIN, MIL_MAX, step=0.5, key="mileage")
                st.slider("Engine (cc)", ENG_MIN, ENG_MAX, step=50, key="engine")
            with s2:
                st.slider("Max power (bhp)", PWR_MIN, PWR_MAX, step=1.0, key="max_power")

            if st.button("↺ Reset to model defaults", use_container_width=True):
                apply_model_defaults()
                st.rerun()

    # ---------- Build Prediction ----------
    input_row = build_input_row(
        st.session_state["model_name"],
        st.session_state["seller_type"],
        st.session_state["fuel_type"],
        st.session_state["transmission_type"],
        st.session_state["vehicle_age"],
        st.session_state["km_driven"],
        st.session_state["mileage"],
        st.session_state["engine"],
        st.session_state["max_power"],
        st.session_state["seats"],
    )

    price, low_est, high_est = predict_with_range(model, input_row)
    model_df = df[df["model"] == st.session_state["model_name"]]
    car_brand = model_df["brand"].iloc[0] if not model_df.empty else ""
    model_median = float(model_df[TARGET].median())
    diff_pct = ((price - model_median) / model_median * 100) if model_median > 0 else 0.0

    # Similar-age median (+/- 1 year)
    sim_age_df = model_df[
        (model_df["vehicle_age"] >= st.session_state["vehicle_age"] - 1)
        & (model_df["vehicle_age"] <= st.session_state["vehicle_age"] + 1)
    ]
    sim_median = float(sim_age_df[TARGET].median()) if not sim_age_df.empty else model_median

    # Range bar positioning
    span_min = min(low_est, float(model_df[TARGET].quantile(0.1)))
    span_max = max(high_est, float(model_df[TARGET].quantile(0.9)))
    denom = max(span_max - span_min, 1.0)
    left_pct = clamp((low_est - span_min) / denom * 100, 0, 95)
    width_pct = clamp((high_est - low_est) / denom * 100, 5, 100 - left_pct)

    with right_col:
        st.markdown(
            f"""
            <div class="price-card">
                <div class="price-label">Estimated Resale Value · {car_brand} {st.session_state['model_name']}</div>
                <div class="price-value">{format_inr(price)}</div>
                <div class="price-sub">
                    {format_lakhs(price)} &nbsp;·&nbsp; Fair market band:
                    <strong>{format_lakhs(low_est)} – {format_lakhs(high_est)}</strong>
                </div>
                <div class="range-track">
                    <div class="range-fill" style="left: {left_pct:.1f}%; width: {width_pct:.1f}%;"></div>
                </div>
                <div class="range-labels">
                    <span>10th pct: {format_lakhs(span_min)}</span>
                    <span>25th–75th IQR</span>
                    <span>90th pct: {format_lakhs(span_max)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Action bar: Save to Compare & Download Report
        act1, act2 = st.columns(2)
        if act1.button("➕ Pin to Compare", use_container_width=True):
            yr3_row = input_row.copy()
            yr3_row["vehicle_age"] = st.session_state["vehicle_age"] + 3
            yr3_row["km_driven"] = st.session_state["km_driven"] + 30_000
            yr3_price = float(model.predict(yr3_row)[0])

            entry = {
                "Car": f"{car_brand} {st.session_state['model_name']}",
                "Age (yrs)": st.session_state["vehicle_age"],
                "KM Driven": st.session_state["km_driven"],
                "Fuel": st.session_state["fuel_type"],
                "Transmission": st.session_state["transmission_type"],
                "Engine (cc)": st.session_state["engine"],
                "Power (bhp)": st.session_state["max_power"],
                "Estimated Price": price,
                "3-Yr Projected": yr3_price,
                "3-Yr Depreciation %": ((yr3_price - price) / price * 100) if price > 0 else 0.0,
            }
            st.session_state["saved_cars"].append(entry)
            st.toast(f"Pinned {entry['Car']} to Compare tab!", icon="📌")
            st.rerun()

        report_text = (
            f"USED CAR VALUATION REPORT\n"
            f"=========================\n"
            f"Vehicle:        {car_brand} {st.session_state['model_name']}\n"
            f"Fuel / Trans:   {st.session_state['fuel_type']} / {st.session_state['transmission_type']}\n"
            f"Seller / Seats: {st.session_state['seller_type']} / {st.session_state['seats']} seats\n"
            f"Age / Mileage:  {st.session_state['vehicle_age']} yrs / {st.session_state['km_driven']:,} km\n"
            f"Specs:          {st.session_state['mileage']} kmpl | {st.session_state['engine']} cc | {st.session_state['max_power']} bhp\n\n"
            f"Estimated Price:   {format_inr(price)} ({format_lakhs(price)})\n"
            f"Fair Market Range: {format_inr(low_est)} - {format_inr(high_est)}\n"
            f"Model Median:      {format_inr(model_median)}\n"
        )
        act2.download_button(
            "📄 Download Valuation",
            data=report_text,
            file_name=f"{st.session_state['model_name'].lower().replace(' ', '_')}_valuation.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # KPI Metrics
        m1, m2, m3, m4 = st.columns(4)
        sim_diff = ((price - sim_median) / sim_median * 100) if sim_median > 0 else 0.0
        m1.metric("Similar-Age Median", format_lakhs(sim_median), f"{sim_diff:+.1f}% vs peers")
        m2.metric("Model Median", format_lakhs(model_median), f"{diff_pct:+.1f}% overall")
        m3.metric("Listings", f"{len(model_df):,}")
        m4.metric("Typical Age", f"{int(model_df['vehicle_age'].median())} yrs")

        # Interactive Insights Tabs
        tab_forecast, tab_drivers, tab_market, tab_comps = st.tabs(
            [
                "📉 Depreciation Forecast",
                "⚡ Price Drivers",
                "📊 Market Position",
                "🔎 Comparable Listings",
            ]
        )

        with tab_forecast:
            fc1, fc2 = st.columns(2)
            annual_km = fc1.slider(
                "Expected annual driving (km/yr)",
                min_value=2000,
                max_value=30000,
                value=10000,
                step=2000,
            )
            horizon_yrs = fc2.slider("Projection horizon (years)", 3, 7, 5)

            future_rows = []
            for yr in range(0, horizon_yrs + 1):
                r = input_row.copy()
                r["vehicle_age"] = st.session_state["vehicle_age"] + yr
                r["km_driven"] = st.session_state["km_driven"] + yr * annual_km
                pred_yr = float(model.predict(r)[0])
                retention = (pred_yr / price * 100) if price > 0 else 100.0
                future_rows.append(
                    {
                        "Year": "Now" if yr == 0 else f"+{yr} yr",
                        "Price (Lakhs)": round(pred_yr / 1e5, 2),
                        "Estimated Value": format_inr(pred_yr),
                        "Retention": f"{retention:.0f}%",
                    }
                )
            forecast_df = pd.DataFrame(future_rows)

            fig_fc = px.area(
                forecast_df,
                x="Year",
                y="Price (Lakhs)",
                markers=True,
                custom_data=["Estimated Value", "Retention"],
            )
            fig_fc.update_traces(
                line=dict(color="#0f766e", width=3),
                fillcolor="rgba(15, 118, 110, 0.18)",
                marker=dict(size=8, color="#0f766e"),
                hovertemplate="<b>%{x}</b><br>Value: %{customdata[0]}<br>Retained: %{customdata[1]}<extra></extra>",
            )
            fig_fc.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title="Price (₹ Lakhs)",
                xaxis_title=None,
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            end_val = forecast_df["Price (Lakhs)"].iloc[-1] * 1e5
            total_drop = max(0.0, price - end_val)
            st.caption(
                f"Projected value after {horizon_yrs} years ({annual_km:,} km/yr): "
                f"**{format_lakhs(end_val)}** (estimated depreciation of **{format_lakhs(total_drop)}**, "
                f"or **{format_lakhs(total_drop / horizon_yrs)}/yr**)."
            )

        with tab_drivers:
            st.caption("How single-factor changes affect this car's predicted resale price:")
            sens_df = compute_sensitivity(model, input_row, price, model_df)
            fig_sens = px.bar(
                sens_df,
                x="Delta",
                y="Factor",
                orientation="h",
                color="Direction",
                text="Delta Label",
                color_discrete_map={
                    "Increases Value": "#0f766e",
                    "Decreases Value": "#e11d48",
                },
            )
            fig_sens.update_traces(
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Impact: %{text}<extra></extra>",
            )
            fig_sens.update_layout(
                height=290,
                margin=dict(l=10, r=30, t=10, b=10),
                xaxis_title="Price Impact (₹)",
                yaxis_title=None,
                showlegend=False,
            )
            fig_sens.add_vline(x=0, line_width=1, line_color="#94a3b8")
            st.plotly_chart(fig_sens, use_container_width=True)

        with tab_market:
            st.caption(
                f"Your {car_brand} {st.session_state['model_name']} vs all "
                f"{len(model_df):,} listings of the same model."
            )
            plot_df = model_df[["vehicle_age", TARGET, "km_driven", "fuel_type", "transmission_type"]].copy()
            plot_df["Price (Lakhs)"] = (plot_df[TARGET] / 1e5).round(2)

            fig_mkt = px.scatter(
                plot_df,
                x="vehicle_age",
                y="Price (Lakhs)",
                color="fuel_type",
                opacity=0.65,
                hover_data=["km_driven", "transmission_type"],
                labels={"vehicle_age": "Vehicle Age (years)", "fuel_type": "Fuel"},
            )
            fig_mkt.add_trace(
                go.Scatter(
                    x=[st.session_state["vehicle_age"]],
                    y=[round(price / 1e5, 2)],
                    mode="markers",
                    name="Your Estimate",
                    marker=dict(
                        size=16,
                        color="#ef4444",
                        symbol="star",
                        line=dict(width=1.5, color="#ffffff"),
                    ),
                    hovertemplate=(
                        f"<b>Your Estimate</b><br>Age: {st.session_state['vehicle_age']} yrs"
                        f"<br>Price: {format_lakhs(price)}<extra></extra>"
                    ),
                )
            )
            fig_mkt.update_layout(
                height=290,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_mkt, use_container_width=True)

        with tab_comps:
            st.caption("Closest real listings in the dataset matched by age, km, and fuel.")
            comps = model_df.copy()
            comps["fuel_match"] = (comps["fuel_type"] != st.session_state["fuel_type"]).astype(int)
            comps["age_diff"] = (comps["vehicle_age"] - st.session_state["vehicle_age"]).abs()
            comps["km_diff"] = (comps["km_driven"] - st.session_state["km_driven"]).abs()
            comps = comps.sort_values(["fuel_match", "age_diff", "km_diff"]).head(8)

            display_comps = pd.DataFrame(
                {
                    "Age (yrs)": comps["vehicle_age"].astype(int),
                    "KM Driven": comps["km_driven"].map(lambda x: f"{int(x):,}"),
                    "Fuel": comps["fuel_type"],
                    "Transmission": comps["transmission_type"],
                    "Seller": comps["seller_type"],
                    "Listed Price": comps[TARGET].map(format_inr),
                    "vs Estimate": comps[TARGET].map(lambda p: f"{((p - price) / price * 100):+.1f}%"),
                }
            )
            st.dataframe(display_comps, hide_index=True, use_container_width=True)

        st.caption(
            f"Estimate only — actual resale prices depend on physical condition, city, and ownership history. "
            f"Model OOB MAE is ≈ {format_inr(OOB_MAE)}."
        )


# ==============================================================================
# TAB 2: COMPARE SAVED CARS
# ==============================================================================
with page_compare:
    st.subheader("⚖️ Side-by-Side Car Comparison")
    if not st.session_state["saved_cars"]:
        st.info(
            "No cars pinned yet. Configure a car in **Price Predictor** and click **➕ Pin to Compare** "
            "to compare multiple cars side-by-side."
        )
        if st.button("Load 3 Popular Cars to Compare"):
            samples = [
                ("Maruti", "Swift", 4, 35000),
                ("Hyundai", "i20", 4, 35000),
                ("Honda", "City", 4, 35000),
            ]
            for br, mod, age_v, km_v in samples:
                sub = df[df["model"] == mod]
                row = build_input_row(
                    mod,
                    sub["seller_type"].mode().iloc[0],
                    sub["fuel_type"].mode().iloc[0],
                    sub["transmission_type"].mode().iloc[0],
                    age_v,
                    km_v,
                    float(round(sub["mileage"].median(), 1)),
                    int(sub["engine"].median()),
                    float(round(sub["max_power"].median(), 1)),
                    int(sub["seats"].mode().iloc[0]),
                )
                p_now = float(model.predict(row)[0])
                row_3 = row.copy()
                row_3["vehicle_age"] += 3
                row_3["km_driven"] += 30_000
                p_3 = float(model.predict(row_3)[0])
                st.session_state["saved_cars"].append(
                    {
                        "Car": f"{br} {mod}",
                        "Age (yrs)": age_v,
                        "KM Driven": km_v,
                        "Fuel": row["fuel_type"].iloc[0],
                        "Transmission": row["transmission_type"].iloc[0],
                        "Engine (cc)": int(row["engine"].iloc[0]),
                        "Power (bhp)": float(row["max_power"].iloc[0]),
                        "Estimated Price": p_now,
                        "3-Yr Projected": p_3,
                        "3-Yr Depreciation %": (p_3 - p_now) / p_now * 100,
                    }
                )
            st.rerun()
    else:
        comp_df = pd.DataFrame(st.session_state["saved_cars"])
        c_top1, c_top2 = st.columns([4, 1])
        c_top1.caption(f"Comparing {len(comp_df)} pinned configurations.")
        if c_top2.button("🗑️ Clear All", use_container_width=True):
            st.session_state["saved_cars"] = []
            st.rerun()

        table_df = comp_df.copy()
        table_df["KM Driven"] = table_df["KM Driven"].map(lambda x: f"{int(x):,}")
        table_df["Estimated Price"] = table_df["Estimated Price"].map(format_inr)
        table_df["3-Yr Projected"] = table_df["3-Yr Projected"].map(format_inr)
        table_df["3-Yr Depreciation %"] = table_df["3-Yr Depreciation %"].map(lambda x: f"{x:+.1f}%")
        st.dataframe(table_df, hide_index=True, use_container_width=True)

        chart_df = comp_df.melt(
            id_vars=["Car"],
            value_vars=["Estimated Price", "3-Yr Projected"],
            var_name="Timeline",
            value_name="Price",
        )
        chart_df["Price (Lakhs)"] = (chart_df["Price"] / 1e5).round(2)
        fig_cmp = px.bar(
            chart_df,
            x="Car",
            y="Price (Lakhs)",
            color="Timeline",
            barmode="group",
            color_discrete_map={
                "Estimated Price": "#0f766e",
                "3-Yr Projected": "#64748b",
            },
        )
        fig_cmp.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_cmp, use_container_width=True)


# ==============================================================================
# TAB 3: MARKET EXPLORER
# ==============================================================================
with page_explore:
    st.subheader("📈 CarDekho Market Explorer")
    st.caption("Explore pricing trends across all 15,411 listings in the dataset.")

    f1, f2, f3, f4 = st.columns(4)
    exp_brands = f1.multiselect("Brands", sorted(df["brand"].unique()), default=[])
    exp_fuels = f2.multiselect("Fuel types", FUELS, default=[])
    exp_trans = f3.multiselect("Transmission", TRANSMISSIONS, default=[])
    max_price_lakhs = int(np.ceil(df[TARGET].quantile(0.99) / 1e5))
    price_range = f4.slider("Price range (₹ Lakhs)", 0, max_price_lakhs, (0, min(30, max_price_lakhs)))

    filtered = df.copy()
    if exp_brands:
        filtered = filtered[filtered["brand"].isin(exp_brands)]
    if exp_fuels:
        filtered = filtered[filtered["fuel_type"].isin(exp_fuels)]
    if exp_trans:
        filtered = filtered[filtered["transmission_type"].isin(exp_trans)]
    filtered = filtered[
        (filtered[TARGET] >= price_range[0] * 1e5)
        & (filtered[TARGET] <= price_range[1] * 1e5)
    ]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Matching Listings", f"{len(filtered):,}")
    if not filtered.empty:
        k2.metric("Median Price", format_lakhs(filtered[TARGET].median()))
        k3.metric("Median Age", f"{int(filtered['vehicle_age'].median())} yrs")
        k4.metric("Median Mileage", f"{filtered['mileage'].median():.1f} kmpl")

        ch1, ch2 = st.columns(2)
        with ch1:
            brand_stats = (
                filtered.groupby("brand", as_index=False)
                .agg(Median_Price=(TARGET, "median"), Listings=(TARGET, "count"))
                .sort_values("Listings", ascending=False)
                .head(12)
            )
            brand_stats["Median (Lakhs)"] = (brand_stats["Median_Price"] / 1e5).round(2)
            fig_b = px.bar(
                brand_stats.sort_values("Median (Lakhs)"),
                x="Median (Lakhs)",
                y="brand",
                orientation="h",
                color="Listings",
                color_continuous_scale="Teal",
                title="Top Brands by Listing Volume & Median Price",
            )
            fig_b.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), yaxis_title=None)
            st.plotly_chart(fig_b, use_container_width=True)

        with ch2:
            sample_n = min(1500, len(filtered))
            samp = filtered.sample(sample_n, random_state=42).copy()
            samp["Price (Lakhs)"] = (samp[TARGET] / 1e5).round(2)
            fig_s = px.scatter(
                samp,
                x="vehicle_age",
                y="Price (Lakhs)",
                color="fuel_type",
                opacity=0.55,
                hover_data=["car_name", "km_driven", "transmission_type"],
                title=f"Price vs Age ({sample_n:,} sampled listings)",
                labels={"vehicle_age": "Vehicle Age (years)", "fuel_type": "Fuel"},
            )
            fig_s.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_s, use_container_width=True)

        with st.expander("View matching listings table (top 100)"):
            view_df = filtered[
                [
                    "car_name",
                    "vehicle_age",
                    "km_driven",
                    "fuel_type",
                    "transmission_type",
                    "mileage",
                    "engine",
                    TARGET,
                ]
            ].head(100).copy()
            view_df[TARGET] = view_df[TARGET].map(format_inr)
            view_df.columns = ["Car", "Age", "KM Driven", "Fuel", "Transmission", "Mileage", "Engine (cc)", "Price"]
            st.dataframe(view_df, hide_index=True, use_container_width=True)
    else:
        st.warning("No listings match the selected filters. Try widening the price range or clearing filters.")

st.divider()
st.caption(
    "Data: CarDekho listings · Model: Random Forest Regressor · "
    "[Source code](https://github.com/Sudip-2005/Car-Price-Prediction)"
)
