import hmac
import hashlib
import os
import re
import sqlite3
import time
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
from fpdf import FPDF

st.set_page_config(
    page_title="Ahmedabad 3D Real Estate AI Valuator",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DB_PATH = os.getenv("DB_PATH", "leads.db")
MODEL_PATH = os.getenv("MODEL_PATH", "house_price_pipeline.pkl")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 30


def _verify_password(entered: str, configured: str) -> bool:
    if not configured:
        return False
    entered_hash = hashlib.sha256(entered.encode("utf-8")).hexdigest()
    configured_hash = hashlib.sha256(configured.encode("utf-8")).hexdigest()
    return hmac.compare_digest(entered_hash, configured_hash)


# --- 1. DATABASE INITIALIZATION & ERROR-SAFE HELPERS ---
def init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS inquiries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT,
                    name TEXT NOT NULL,
                    phone TEXT NOT NULL,
                    email TEXT,
                    inquiry_type TEXT,
                    location TEXT,
                    property_type TEXT,
                    area_sqft REAL,
                    bhk INTEGER,
                    estimated_price TEXT,
                    message TEXT,
                    status TEXT DEFAULT 'New'
                )
            """)
            conn.commit()
    except Exception as e:
        st.error(f"Database initialization warning: {e}")


def save_inquiry(data: dict) -> bool:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO inquiries (
                    created_at, name, phone, email, inquiry_type,
                    location, property_type, area_sqft, bhk, estimated_price, message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data["name"], data["phone"], data["email"], data["inquiry_type"],
                data["location"], data["property_type"], data["area_sqft"],
                data["bhk"], data["estimated_price"], data["message"]
            ))
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Could not save inquiry: {e}")
        return False


def fetch_inquiries() -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query("SELECT * FROM inquiries ORDER BY id DESC", conn)
    except Exception:
        return pd.DataFrame()


def update_lead_status(lead_id: int, new_status: str):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("UPDATE inquiries SET status = ? WHERE id = ?", (new_status, lead_id))
            conn.commit()
    except Exception as e:
        st.error(f"Status update failed: {e}")


init_db()


# --- 2. PDF VALUATION REPORT GENERATOR ---
def generate_pdf_report(details: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header Banner
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 38, "F")
    pdf.set_text_color(56, 189, 248)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(14, 10)
    pdf.cell(0, 8, "AHMEDABAD PROPERTY VALUATION AI", ln=True)
    pdf.set_text_color(203, 213, 225)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(14)
    pdf.cell(0, 6, f"Official AI-Generated Market Valuation Certificate  |  Report ID: #{details['report_id']}", ln=True)

    # Valuation Highlight Box
    pdf.set_y(46)
    pdf.set_fill_color(240, 249, 255)
    pdf.set_draw_color(14, 165, 233)
    pdf.rect(14, 46, 182, 34, "DF")
    pdf.set_xy(18, 50)
    pdf.set_text_color(71, 85, 105)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 5, "ESTIMATED FAIR MARKET VALUE", ln=True)
    pdf.set_x(18)
    pdf.set_text_color(2, 132, 199)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 10, f"INR {details['main_price_ascii']}   ({details['alt_price_ascii']})", ln=True)
    pdf.set_x(18)
    pdf.set_text_color(51, 65, 85)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Recommended Market Band: {details['range_ascii']}   |   Avg. Rate: INR {details['rate_sqft']:,.0f} / sq.ft.", ln=True)

    # Property Specifications Table
    pdf.set_y(88)
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Property Specifications & Location Profile", ln=True)
    pdf.ln(2)

    rows = [
        ("Valuation Date & Time", details["timestamp"]),
        ("Micro-Market / Project", details["location"]),
        ("Property Category", details["property_type"]),
        ("Area Measurement Type", details["area_type"]),
        ("Total Super / Plot Area", f"{details['area_sqft']:,} Sq.Ft."),
        ("Configuration", "Open Plot / Land" if details["bhk"] == 0 else f"{details['bhk']} BHK"),
        ("Model Fit (R2) & MAE", f"{details['r2_score']}%  |  +/- INR {details['mae_cr']:.2f} Cr ({details['samples']:,} listings)")
    ]

    for label, val in rows:
        pdf.set_fill_color(248, 250, 252)
        pdf.set_draw_color(226, 232, 240)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(75, 9, f"  {label}", border=1, fill=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(107, 9, f"  {val}", border=1, ln=True)

    # Legal & Methodology Note
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, "2. Methodology & Legal Disclaimer", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.multi_cell(
        0, 5,
        "This automated valuation is generated using an XGBoost Machine Learning regression pipeline calibrated on historical "
        "Ahmedabad residential real estate market listings. Actual transaction prices may vary based on floor rise, furnishing, "
        "exact title clearance, builder reputation, and RERA compliance. This certificate is intended for informational "
        "and advisory purposes and does not constitute a statutory bank or government valuation."
    )

    pdf.set_y(-22)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 6, "Generated by Ahmedabad Property Valuation AI  |  Ahmedabad, Gujarat, India", align="C")

    return bytes(pdf.output())


# --- 3. STYLING & LUXURY RESPONSIVE THEME ---
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at 15% 15%, #1e1b4b 0%, #090d16 55%, #020617 100%);
        color: #ffffff;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    .block-container {
        padding-top: 1.3rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 1280px !important;
    }
    /* Scope the glassmorphic configurator card strictly to Tab 1's left column */
    div[data-baseweb="tab-panel"]:first-of-type div[data-testid="stHorizontalBlock"] > div:first-child {
        background: rgba(15, 23, 42, 0.78);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-radius: 20px;
        padding: 22px !important;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.6);
    }
    label, [data-testid="stWidgetLabel"] p {
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }
    .hero-wrapper {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 0.2rem;
    }
    .hero-icon {
        width: 40px;
        height: 40px;
        flex-shrink: 0;
        filter: drop-shadow(0 0 12px rgba(56, 189, 248, 0.6));
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #e879f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.96rem;
        margin-bottom: 1rem;
    }
    .result-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.18), rgba(139, 92, 246, 0.22));
        border: 1px solid rgba(56, 189, 248, 0.55);
        border-radius: 16px;
        padding: 18px 22px;
        margin-top: 10px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        backdrop-filter: blur(14px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
    }
    .result-label {
        color: #cbd5e1;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .result-price {
        color: #38bdf8;
        font-size: 2.15rem;
        font-weight: 800;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.45);
        line-height: 1.15;
    }
    .result-range {
        color: #7dd3fc;
        font-size: 0.84rem;
        margin-top: 3px;
    }
    .result-sub {
        color: #e2e8f0;
        font-size: 0.9rem;
        text-align: right;
    }
    .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
        width: 100%;
        background: linear-gradient(90deg, #0284c7, #6366f1);
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.2rem;
    }
    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 1.5rem;
    }
    @media (max-width: 768px) {
        .hero-title { font-size: 1.5rem; }
        .result-card { flex-direction: column; align-items: flex-start; }
        .result-sub { text-align: left; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px; width: 100%; }
    }
    </style>
""", unsafe_allow_html=True)


# --- 4. LOAD ML ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


artifacts = load_artifacts()
if artifacts is None:
    st.error("⚠️ Model artifact `house_price_pipeline.pkl` missing or unreadable. Run `python train_model.py` first.")
    st.stop()

model = artifacts["model"]
locations = artifacts["locations"]
property_types = artifacts["property_types"]
area_types = artifacts.get("area_types", ["Super", "Carpet", "Built-Up", "Plot"])
metrics = artifacts.get("metrics", {"r2": 85.0, "mae": 0.18, "samples": 5000})
model_mae = float(metrics.get("mae", 0.18))

# --- 5. HERO HEADER & NAVIGATION TABS ---
st.markdown("""
    <div class="hero-wrapper">
        <svg class="hero-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 10.5L12 3L21 10.5V20C21 20.5523 20.5523 21 20 21H15V14H9V21H4C3.44772 21 3 20.5523 3 20V10.5Z" fill="url(#houseGrad)" stroke="#7dd3fc" stroke-width="1.5" stroke-linejoin="round"/>
            <defs>
                <linearGradient id="houseGrad" x1="3" y1="3" x2="21" y2="21" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#38bdf8"/>
                    <stop offset="1" stop-color="#c084fc"/>
                </linearGradient>
            </defs>
        </svg>
        <div class="hero-title">Ahmedabad Property Valuation AI</div>
    </div>
""", unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">3D Architectural Configurator, Instant Valuation Certificate & Advisory Portal • Ahmedabad</div>', unsafe_allow_html=True)

tab_valuator, tab_inquiry, tab_admin, tab_legal = st.tabs([
    "🏙️ 3D AI Valuator & PDF Report",
    "📞 Book Consultation / Inquiry",
    "🔐 Admin Dashboard",
    "📜 Privacy Policy & Terms"
])

# ==========================================
# TAB 1: 3D VALUATOR + DOWNLOADABLE PDF
# ==========================================
with tab_valuator:
    left_col, right_col = st.columns([1.0, 1.4], gap="large")

    with left_col:
        st.markdown("### 🎛️ Configure Property")
        location = st.selectbox("📍 Location / Project", locations)
        property_type = st.selectbox("🏢 Property Type", property_types)
        area_type = st.selectbox("📏 Area Basis", area_types)
        area_in_sqft = st.number_input("📐 Total Area (Sq.Ft.)", min_value=250, max_value=15000, value=1400, step=50)

        prop_lower = str(property_type).lower()
        is_land = any(word in prop_lower for word in ["land", "plot"])
        is_floor = "floor" in prop_lower
        is_house = any(word in prop_lower for word in ["house", "villa", "bungalow", "independent"]) and not is_floor

        if is_land:
            st.info("🌱 Open Land/Plot selected — BHK configuration is disabled.")
            bhk_type = 0
        else:
            bhk_type = st.slider("🛏️ Configuration (BHK)", min_value=1, max_value=6, value=3)

        st.caption(f"📊 Model Fit (R²): **{metrics['r2']}%** | Avg Error (MAE): **±₹{model_mae:.2f} Cr** | **{metrics['samples']:,}** listings")

    # Safe Prediction Execution (no silent fake price fallback)
    prediction_ok = True
    pred_cr = None
    try:
        input_row = {
            "location": location,
            "area_in_sqft": float(area_in_sqft),
            "area_type": area_type,
            "bhk_type": int(bhk_type),
            "property_type": property_type
        }
        input_df = pd.DataFrame([input_row])
        pred_cr = max(0.08, float(model.predict(input_df)[0]))
    except Exception as e:
        prediction_ok = False
        st.error(f"⚠️ Unable to compute valuation for the selected inputs: {e}")

    scene_mode = "land" if is_land else ("floor" if is_floor else ("house" if is_house else "flat"))
    scale_factor = min(1.45, max(0.85, area_in_sqft / 1600.0))
    badge_text = f"OPEN LAND • {area_in_sqft} SQ.FT." if is_land else f"{property_type.upper()} • {bhk_type} BHK • {area_in_sqft} SQ.FT."

    with right_col:
        three_js_html = f"""
        <div id="canvas-container" style="width: 100%; height: 325px; border-radius: 20px; overflow: hidden; border: 1px solid rgba(56, 189, 248, 0.35); background: linear-gradient(180deg, #0c192c 0%, #17253b 65%, #0f172a 100%); position: relative; box-shadow: 0 20px 45px rgba(0,0,0,0.65); cursor: grab; touch-action: none;">
            <div style="position: absolute; top: 12px; left: 12px; right: 95px; background: rgba(15,23,42,0.82); padding: 6px 10px; border-radius: 8px; border: 1px solid rgba(56,189,248,0.35); color: #38bdf8; font-family: sans-serif; font-size: 11px; font-weight: 700; letter-spacing: 0.6px; z-index: 10; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                🏛️ {badge_text}
            </div>
            <button id="toggle-spin" style="position: absolute; top: 12px; right: 12px; background: rgba(15,23,42,0.85); color: #f8fafc; border: 1px solid rgba(56,189,248,0.5); border-radius: 8px; padding: 5px 9px; font-size: 11px; font-weight: 600; cursor: pointer; z-index: 10;">
                ⏸ Pause
            </button>
            <div style="position: absolute; bottom: 10px; left: 12px; color: #cbd5e1; font-family: sans-serif; font-size: 11px; z-index: 10; background: rgba(15,23,42,0.7); padding: 4px 10px; border-radius: 6px;">
                🖱️/👆 Drag to Rotate 360° &nbsp;•&nbsp; 🔍 Scroll to Zoom
            </div>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
            const container = document.getElementById('canvas-container');
            const scene = new THREE.Scene();
            scene.fog = new THREE.FogExp2(0x111c2e, 0.045);
            const camera = new THREE.PerspectiveCamera(42, container.clientWidth / container.clientHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.shadowMap.enabled = true;
            container.appendChild(renderer.domElement);

            const ground = new THREE.Mesh(new THREE.CylinderGeometry(5.2, 5.2, 0.18, 48), new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.9 }}));
            ground.position.y = -1.2;
            scene.add(ground);

            const road = new THREE.Mesh(new THREE.BoxGeometry(6.8, 0.04, 1.1), new THREE.MeshStandardMaterial({{ color: 0x334155, roughness: 0.85 }}));
            road.position.set(0, -1.09, 2.5);
            scene.add(road);

            function createTree(x, z, scale=1) {{
                const tree = new THREE.Group();
                const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.06*scale, 0.09*scale, 0.55*scale, 8), new THREE.MeshStandardMaterial({{ color: 0x5c4033 }}));
                trunk.position.y = -0.85;
                tree.add(trunk);
                const c1 = new THREE.Mesh(new THREE.DodecahedronGeometry(0.32*scale, 1), new THREE.MeshStandardMaterial({{ color: 0x2d6a4f }}));
                c1.position.y = -0.45;
                tree.add(c1);
                tree.position.set(x, 0, z);
                return tree;
            }}

            function createWindow(w, h, d, lit=true) {{
                return new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshStandardMaterial({{
                    color: lit ? 0xfef08a : 0x38bdf8,
                    emissive: lit ? 0xf59e0b : 0x0284c7,
                    emissiveIntensity: lit ? 0.55 : 0.25,
                    roughness: 0.15, metalness: 0.7
                }}));
            }}

            const propGroup = new THREE.Group();
            const mode = "{scene_mode}";
            const s = {scale_factor};
            const bhk = {bhk_type};

            if (mode === "land") {{
                const plotW = 2.8 * s, plotD = 2.3 * s;
                const grass = new THREE.Mesh(new THREE.BoxGeometry(plotW, 0.12, plotD), new THREE.MeshStandardMaterial({{ color: 0x386641 }}));
                grass.position.y = -1.05;
                propGroup.add(grass);
                const wallMat = new THREE.MeshStandardMaterial({{ color: 0xd6ccc2 }});
                const bw = new THREE.Mesh(new THREE.BoxGeometry(plotW, 0.35, 0.1), wallMat); bw.position.set(0, -0.88, -plotD/2); propGroup.add(bw);
                const lw = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.35, plotD), wallMat); lw.position.set(-plotW/2, -0.88, 0); propGroup.add(lw);
                const rw = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.35, plotD), wallMat); rw.position.set(plotW/2, -0.88, 0); propGroup.add(rw);
                propGroup.add(createTree(-plotW/2 + 0.35, -plotD/2 + 0.35, 0.9));
                propGroup.add(createTree(plotW/2 - 0.35, -plotD/2 + 0.4, 0.75));
            }} else if (mode === "house") {{
                const w = 1.8 * s, d = 1.45 * s;
                const lawn = new THREE.Mesh(new THREE.BoxGeometry(w + 1.3, 0.1, d + 1.3), new THREE.MeshStandardMaterial({{ color: 0x386641 }}));
                lawn.position.y = -1.06; propGroup.add(lawn);
                const stories = bhk >= 4 ? 2 : 1;
                const wallMat = new THREE.MeshStandardMaterial({{ color: 0xf5ebe0 }});
                for (let i = 0; i < stories; i++) {{
                    const yPos = -0.62 + (i * 0.76);
                    const body = new THREE.Mesh(new THREE.BoxGeometry(w, 0.76, d), wallMat);
                    body.position.y = yPos; propGroup.add(body);
                    const win1 = createWindow(0.42, 0.34, 0.06, true);
                    win1.position.set(-w * 0.26, yPos + 0.04, d/2 + 0.02); propGroup.add(win1);
                }}
                const roof = new THREE.Mesh(new THREE.ConeGeometry(w * 0.86, 0.62, 4), new THREE.MeshStandardMaterial({{ color: 0x9e2a2b }}));
                roof.position.y = -0.62 + (stories * 0.76) - 0.05; roof.rotation.y = Math.PI / 4; propGroup.add(roof);
                propGroup.add(createTree(-w/2 - 0.45, -0.4, 0.95));
            }} else if (mode === "floor") {{
                const w = 1.65 * s, d = 1.35 * s;
                const stilt = new THREE.Mesh(new THREE.BoxGeometry(w, 0.45, d), new THREE.MeshStandardMaterial({{ color: 0x475569 }}));
                stilt.position.y = -0.88; propGroup.add(stilt);
                for (let i = 0; i < 3; i++) {{
                    const yPos = -0.42 + (i * 0.52);
                    const sel = (i === Math.min(2, Math.floor(bhk / 2)));
                    const fb = new THREE.Mesh(new THREE.BoxGeometry(w, 0.46, d), new THREE.MeshStandardMaterial({{ color: sel ? 0xe0f2fe : 0xe2e8f0 }}));
                    fb.position.y = yPos; propGroup.add(fb);
                    const win = createWindow(w * 0.55, 0.24, 0.05, sel);
                    win.position.set(0, yPos + 0.04, d/2 + 0.02); propGroup.add(win);
                }}
                propGroup.add(createTree(-w/2 - 0.5, 0.4, 0.9));
            }} else {{
                const floors = Math.max(4, bhk + 3);
                const w = 1.45 * s, d = 1.25 * s;
                const podium = new THREE.Mesh(new THREE.BoxGeometry(w + 0.9, 0.25, d + 0.9), new THREE.MeshStandardMaterial({{ color: 0x334155 }}));
                podium.position.y = -0.98; propGroup.add(podium);
                const facadeMat = new THREE.MeshStandardMaterial({{ color: 0xcbd5e1 }});
                for (let i = 0; i < floors; i++) {{
                    const yPos = -0.68 + (i * 0.36);
                    const fb = new THREE.Mesh(new THREE.BoxGeometry(w, 0.32, d), facadeMat);
                    fb.position.y = yPos; propGroup.add(fb);
                    [-w * 0.26, w * 0.26].forEach((wx, idx) => {{
                        const fw = createWindow(w * 0.34, 0.18, 0.04, (i + idx) % 2 === 0);
                        fw.position.set(wx, yPos, d/2 + 0.01); propGroup.add(fw);
                    }});
                }}
                propGroup.add(createTree(-w/2 - 0.55, 0.45, 0.85));
            }}

            scene.add(propGroup);
            scene.add(new THREE.HemisphereLight(0xe0f2fe, 0x1e293b, 0.85));
            const sun = new THREE.DirectionalLight(0xfffbeb, 1.25); sun.position.set(5, 9, 6); scene.add(sun);
            camera.position.set(4.1, 2.5, 5.1); camera.lookAt(0, -0.1, 0);

            let isDragging = false, autoSpin = true, prevX = 0, prevY = 0;
            document.getElementById('toggle-spin').addEventListener('click', (e) => {{
                autoSpin = !autoSpin; e.target.innerText = autoSpin ? "⏸ Pause" : "▶ Spin";
            }});
            const start = (x, y) => {{ isDragging = true; prevX = x; prevY = y; }};
            const move = (x, y) => {{ if (!isDragging) return; propGroup.rotation.y += (x - prevX) * 0.012; propGroup.rotation.x = Math.max(-0.35, Math.min(0.45, propGroup.rotation.x + (y - prevY) * 0.008)); prevX = x; prevY = y; }};
            container.addEventListener('mousedown', (e) => start(e.clientX, e.clientY));
            window.addEventListener('mouseup', () => isDragging = false);
            window.addEventListener('mousemove', (e) => move(e.clientX, e.clientY));
            container.addEventListener('touchstart', (e) => {{ if (e.touches.length === 1) start(e.touches[0].clientX, e.touches[0].clientY); }}, {{ passive: true }});
            container.addEventListener('touchmove', (e) => {{ if (e.touches.length === 1) move(e.touches[0].clientX, e.touches[0].clientY); }}, {{ passive: true }});
            container.addEventListener('touchend', () => isDragging = false);
            container.addEventListener('wheel', (e) => {{ e.preventDefault(); camera.position.z = Math.max(2.8, Math.min(9.0, camera.position.z + e.deltaY * 0.004)); }});
            window.addEventListener('resize', () => {{
                if (!container.clientWidth) return;
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }});
            function animate() {{ requestAnimationFrame(animate); if (!isDragging && autoSpin) propGroup.rotation.y += 0.006; renderer.render(scene, camera); }}
            animate();
        </script>
        """
        components.html(three_js_html, height=335)

        if prediction_ok and pred_cr is not None:
            pred_lakhs = pred_cr * 100
            rate_per_sqft = (pred_cr * 10000000) / area_in_sqft
            low_cr = max(0.08, pred_cr - model_mae)
            high_cr = pred_cr + model_mae

            main_price = f"₹ {pred_cr:.2f} Cr" if pred_cr >= 1.0 else f"₹ {pred_lakhs:.2f} Lakhs"
            alt_price = f"≈ ₹ {pred_lakhs:,.0f} Lakhs" if pred_cr >= 1.0 else f"≈ ₹ {pred_cr:.2f} Cr"
            range_text = (
                f"Est. Market Band (±MAE): ₹ {low_cr:.2f} Cr – ₹ {high_cr:.2f} Cr"
                if pred_cr >= 1.0
                else f"Est. Market Band (±MAE): ₹ {low_cr*100:.1f} L – ₹ {high_cr*100:.1f} L"
            )

            main_price_ascii = f"{pred_cr:.2f} Cr" if pred_cr >= 1.0 else f"{pred_lakhs:.2f} Lakhs"
            alt_price_ascii = f"Approx. INR {pred_lakhs:,.0f} Lakhs" if pred_cr >= 1.0 else f"Approx. INR {pred_cr:.2f} Cr"
            range_ascii = (
                f"INR {low_cr:.2f} Cr - {high_cr:.2f} Cr"
                if pred_cr >= 1.0
                else f"INR {low_cr*100:.1f} L - {high_cr*100:.1f} L"
            )

            st.session_state["last_valuation"] = {
                "location": location,
                "property_type": property_type,
                "area_sqft": area_in_sqft,
                "bhk": bhk_type,
                "estimated_price": main_price
            }

            st.markdown(f"""
                <div class="result-card">
                    <div>
                        <div class="result-label">Live Market Valuation</div>
                        <div class="result-price">{main_price}</div>
                        <div class="result-range">{range_text}</div>
                    </div>
                    <div class="result-sub">
                        <div>{alt_price}</div>
                        <div style="color: #94a3b8; margin-top: 3px;">Avg. Rate: <b style="color:#f8fafc;">₹ {rate_per_sqft:,.0f} / sq.ft.</b></div>
                        <div style="color: #64748b; font-size: 0.8rem; margin-top: 2px;">{location}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            pdf_bytes = generate_pdf_report({
                "report_id": datetime.now().strftime("%Y%m%d%H%M"),
                "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                "location": location,
                "property_type": property_type,
                "area_type": area_type,
                "area_sqft": area_in_sqft,
                "bhk": bhk_type,
                "main_price_ascii": main_price_ascii,
                "alt_price_ascii": alt_price_ascii,
                "range_ascii": range_ascii,
                "rate_sqft": rate_per_sqft,
                "r2_score": metrics["r2"],
                "mae_cr": model_mae,
                "samples": metrics["samples"]
            })

            st.download_button(
                label="📄 Download Official PDF Valuation Report",
                data=pdf_bytes,
                file_name=f"Ahmedabad_Valuation_{re.sub(r'[^A-Za-z0-9]+', '_', location)[:20]}.pdf",
                mime="application/pdf"
            )

# ==========================================
# TAB 2: CUSTOMER INQUIRY / CONTACT FORM
# ==========================================
with tab_inquiry:
    st.markdown("### 📞 Request Expert Property Consultation")
    st.markdown("Want to buy, sell, schedule a site visit, or get a bank-approved valuation in Ahmedabad? Submit your details below.")

    val_ctx = st.session_state.get("last_valuation", {})
    st.info(f"🔗 **Attached Valuation Context:** {val_ctx.get('property_type', 'Flat')} in **{val_ctx.get('location', 'Ahmedabad')}** ({val_ctx.get('area_sqft', 1400)} sq.ft.) — Estimated at **{val_ctx.get('estimated_price', 'N/A')}**")

    with st.form("customer_inquiry_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            cust_name = st.text_input("Full Name *", placeholder="e.g., Aarav Patel")
            cust_phone = st.text_input("Mobile Number (10 digits) *", placeholder="e.g., 9876543210")
        with c2:
            cust_email = st.text_input("Email Address (Optional)", placeholder="e.g., aarav@example.com")
            inquiry_type = st.selectbox("Inquiry Purpose", [
                "Buy Property in This Area",
                "Sell My Property (Get Best Offer)",
                "Schedule Physical Site Visit",
                "Home Loan & Legal Verification"
            ])
        cust_msg = st.text_area("Additional Requirements / Preferred Time", placeholder="Tell us your budget range, timeline, or preferred call time...")
        consent = st.checkbox("I agree to the Privacy Policy & Terms of Use and consent to be contacted regarding my inquiry.")

        submit_lead = st.form_submit_button("🚀 Submit Property Inquiry")

        if submit_lead:
            clean_phone = re.sub(r"\D", "", cust_phone)
            if len(cust_name.strip()) < 2:
                st.error("❌ Please enter a valid full name.")
            elif len(clean_phone) < 10:
                st.error("❌ Please enter a valid 10-digit Indian mobile number.")
            elif cust_email.strip() and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", cust_email.strip()):
                st.error("❌ Please enter a valid email address.")
            elif not consent:
                st.warning("⚠️ Please accept the Privacy Policy & Terms before submitting.")
            else:
                saved = save_inquiry({
                    "name": cust_name.strip(),
                    "phone": clean_phone[-10:],
                    "email": cust_email.strip(),
                    "inquiry_type": inquiry_type,
                    "location": val_ctx.get("location", "N/A"),
                    "property_type": val_ctx.get("property_type", "N/A"),
                    "area_sqft": val_ctx.get("area_sqft", 0),
                    "bhk": val_ctx.get("bhk", 0),
                    "estimated_price": val_ctx.get("estimated_price", "N/A"),
                    "message": cust_msg.strip()
                })
                if saved:
                    st.success("✅ Thank you! Your inquiry has been registered. Our Ahmedabad property advisor will contact you shortly.")

# ==========================================
# TAB 3: ADMIN DASHBOARD (PASSWORD PROTECTED)
# ==========================================
with tab_admin:
    st.markdown("### 🔐 Advisor & Admin CRM Dashboard")

    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False
    if "admin_failed_attempts" not in st.session_state:
        st.session_state["admin_failed_attempts"] = 0
    if "admin_lockout_until" not in st.session_state:
        st.session_state["admin_lockout_until"] = 0.0

    if not ADMIN_PASSWORD:
        st.warning("🔒 Admin portal is locked. Set the `ADMIN_PASSWORD` environment variable on the server to enable admin login.")
    elif not st.session_state["admin_authenticated"]:
        remaining = int(st.session_state["admin_lockout_until"] - time.time())
        if remaining > 0:
            st.error(f"⏳ Too many failed attempts. Try again in {remaining} seconds.")
        else:
            pwd = st.text_input("Enter Admin Passcode", type="password", placeholder="Enter your configured admin password")
            if st.button("Unlock Admin Portal"):
                if _verify_password(pwd, ADMIN_PASSWORD):
                    st.session_state["admin_authenticated"] = True
                    st.session_state["admin_failed_attempts"] = 0
                    st.rerun()
                else:
                    st.session_state["admin_failed_attempts"] += 1
                    if st.session_state["admin_failed_attempts"] >= MAX_LOGIN_ATTEMPTS:
                        st.session_state["admin_lockout_until"] = time.time() + LOCKOUT_SECONDS
                        st.session_state["admin_failed_attempts"] = 0
                        st.error(f"❌ Too many incorrect attempts. Locked for {LOCKOUT_SECONDS}s.")
                    else:
                        st.error("❌ Incorrect passcode.")
    else:
        col_top1, col_top2 = st.columns([4, 1])
        with col_top1:
            st.success("Authenticated as Admin • Live CRM & Model Analytics")
        with col_top2:
            if st.button("🚪 Logout"):
                st.session_state["admin_authenticated"] = False
                st.rerun()

        if DB_PATH == "leads.db":
            st.info("💾 Currently storing leads in local SQLite (`leads.db`). For cloud hosting, mount persistent storage or configure a managed database.")

        leads_df = fetch_inquiries()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Inquiries", len(leads_df))
        k2.metric("New Leads", int((leads_df["status"] == "New").sum()) if not leads_df.empty else 0)
        k3.metric("Model Fit (R²)", f"{metrics['r2']}%")
        k4.metric("Dataset Size", f"{metrics['samples']:,} Listings")

        st.divider()

        if leads_df.empty:
            st.info("No customer inquiries submitted yet. Test submitting one from the 'Book Consultation / Inquiry' tab!")
        else:
            st.markdown("#### 📋 Customer Inquiries & Lead Management")
            st.dataframe(leads_df, use_container_width=True, hide_index=True)

            c_upd1, c_upd2, c_upd3 = st.columns([1.2, 1.5, 1.3])
            with c_upd1:
                sel_id = st.selectbox("Select Inquiry ID", leads_df["id"].tolist())
            with c_upd2:
                new_st = st.selectbox("Update Status", ["New", "Contacted", "Site Visit Scheduled", "Closed"])
            with c_upd3:
                st.write("")
                st.write("")
                if st.button("Update Lead Status"):
                    update_lead_status(int(sel_id), new_st)
                    st.success(f"Lead #{sel_id} updated to '{new_st}'!")
                    st.rerun()

            csv_data = leads_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export All Leads as CSV",
                data=csv_data,
                file_name=f"ahmedabad_leads_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# ==========================================
# TAB 4: PRIVACY POLICY & TERMS OF USE
# ==========================================
with tab_legal:
    st.markdown("### 📜 Privacy Policy & Terms of Service")
    st.markdown("""
    **Effective Date:** November 2026  
    **Platform:** Ahmedabad Property Valuation AI (Gujarat, India)

    #### 1. Privacy Policy & Data Protection
    * **Information We Collect:** When you submit a property consultation request, we collect your name, phone number, email address (optional), and selected property parameters (location, area, property type, and estimated valuation).
    * **How We Use Your Data:** Your contact details are strictly used to connect you with verified Ahmedabad real estate advisors for site visits, valuation verification, or home loan assistance. **We never sell or rent your personal data to third-party marketing agencies.**
    * **Data Security & Retention:** Inquiry records are stored securely in an access-controlled database accessible only to authorized administrators. You may request deletion of your contact records at any time.

    #### 2. Terms of Use & Valuation Disclaimer
    * **Indicative AI Estimates:** All property prices, per-sq.ft. rates, and market bands generated by this tool are statistical estimates derived from historical Ahmedabad real estate market listings using an XGBoost Machine Learning model.
    * **Not a Legal Appraisal:** This platform does not provide statutory bank appraisals, RERA legal clearances, or registered government valuations. Users should conduct independent legal and physical due diligence before executing any real estate transaction.
    * **3D Architectural Previews:** The interactive 3D visualizations represent conceptual architectural massing based on user-selected property type and area, and do not depict exact floor plans of any specific builder project.
    """)

st.markdown('<div class="footer-note">© 2026 Ahmedabad Property Valuation AI • Powered by XGBoost Machine Learning • Privacy & Terms Apply</div>', unsafe_allow_html=True)