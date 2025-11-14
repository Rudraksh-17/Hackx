<<<<<<< HEAD
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime

# NEW: AI audit deps
import cv2
import numpy as np

# ==================== DATABASE SETUP ====================
conn = sqlite3.connect("aid_system.db", check_same_thread=False)
cur = conn.cursor()

cur.execute('''CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT
)''')

cur.execute('''CREATE TABLE IF NOT EXISTS aid_records(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    donor TEXT,
    beneficiary TEXT,
    amount REAL,
    date TEXT,
    status TEXT
)''')

conn.commit()

# ==================== PAGE SETTINGS & GLOBAL STYLE ====================
st.set_page_config(
    page_title="Aid Auditing Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.stApp { background: radial-gradient(1200px 600px at 20% 0%, #1d2733 0%, #11161b 40%, #0a0f14 100%) !important; }
h1, h2, h3, .main-title { color: #e6f1ff; letter-spacing: .2px; }
/* Hero */
.hero { border-radius: 20px; padding: 28px; background: linear-gradient(145deg, rgba(15,21,32,.9), rgba(18,26,38,.9));
         border: 1px solid rgba(255,255,255,.06); box-shadow: 0 16px 36px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04); }
.hero h1 { margin: 0 0 6px 0; font-size: 1.8rem; }
.sub { color:#9fb4c7; }
/* Metric cards */
.metric-card { border-radius: 16px; padding: 18px 18px 12px; background: linear-gradient(145deg, #0f1520, #121a26);
               border: 1px solid rgba(255,255,255,.06); box-shadow: 0 10px 24px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04); }
.metric-label { color: #8ca3b8; font-size: .85rem; }
.metric-value { color: #e8f2ff; font-size: 1.6rem; font-weight: 700; }
/* Chips */
.chip { display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px;
        background:rgba(0,191,255,.12); color:#aee4ff; font-size:.85rem; border:1px solid rgba(0,191,255,.25); }
/* Tables */
.dataframe thead tr th { background: #0e141c !important; color:#adc1d1 !important; }
</style>
""", unsafe_allow_html=True)

# ==================== HELPERS ====================
def add_user(username, password, role="user"):
    try:
        cur.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
        conn.commit()
        return True
    except Exception:
        return False

def verify_user(username, password):
    cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return cur.fetchone()

@st.cache_data(ttl=10)
def get_aid_records_cached():
    return pd.read_sql("SELECT * FROM aid_records", conn)

def refresh_cache():
    get_aid_records_cached.clear()

def add_aid_record(donor, beneficiary, amount, status, date_str=None):
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    cur.execute(
        "INSERT INTO aid_records (donor, beneficiary, amount, date, status) VALUES (?, ?, ?, ?, ?)",
        (donor, beneficiary, amount, date_str, status)
    )
    conn.commit()
    refresh_cache()

def delete_records(record_ids):
    cur.executemany("DELETE FROM aid_records WHERE id=?", [(int(rid),) for rid in record_ids])
    conn.commit()
    refresh_cache()

def human_currency(x):
    try:
        return f"${x:,.2f}"
    except Exception:
        return "$0.00"

# ==================== AI IMAGE AUDIT (INTEGRATED) ====================
# Config
HAMMING_THRESHOLD = 5

def calculate_dhash(image_bytes, hash_size=8):
    """Compute dHash from raw image bytes (safe for Streamlit uploader)."""
    try:
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Invalid image data. Please upload a valid JPG/PNG.")
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        diff = resized[:, :-1] > resized[:, 1:]
        # pack bits into a 64-bit integer
        dhash = 0
        for i, bit in enumerate(diff.flatten()):
            if bit:
                dhash |= (1 << i)
        return np.uint64(dhash)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def hamming_distance(hash1, hash2):
    """Portable Hamming distance (works for numpy uint64)."""
    return int(int(hash1) ^ int(hash2)).bit_count()

@st.cache_data
def generate_mock_hashes():
    """Deterministic mock hashes simulating ledger records."""
    ORIGINAL_HASH_MOCK = np.uint64(0x40808183878F9FBF)
    UNIQUE_HASH_MOCK = np.uint64(0xFFFFFFFFFFFFFFFF)
    return [
        ("Recorded Photo #1 (Original)", ORIGINAL_HASH_MOCK),
        ("Recorded Photo #2 (Unique)", UNIQUE_HASH_MOCK),
    ]

def audit_submission(new_hash, existing_hashes):
    """Run near-duplicate check and display results."""
    st.info(f"New Submission Hash (dHash): `{hex(int(new_hash))}`")
    flagged = False
    for description, existing_hash in existing_hashes:
        distance = hamming_distance(new_hash, existing_hash)
        st.caption(f"Comparing with existing record **{description}** …")
        st.text(f"  → Hamming Distance: {distance}")
        if distance <= HAMMING_THRESHOLD:
            st.error(f"🚫 AUDIT FAILED: Near-duplicate detected (distance {distance} ≤ {HAMMING_THRESHOLD}).")
            flagged = True
            break
    if not flagged:
        st.success("✅ AUDIT SUCCESS: Photo appears unique. Ready for blockchain logging.")
    return not flagged

# A modal popup for the whole audit flow (requires Streamlit ≥ 1.31)
def render_ai_audit_dialog():
    existing_records = generate_mock_hashes()
    uploaded_file = st.file_uploader(
        "Upload Field Worker Submission Photo (JPG or PNG)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.image(uploaded_file, caption="Submitted Photo", use_column_width=True)
        with c2:
            st.subheader("Audit Report")
            if st.button("Run Image Audit", use_container_width=True):
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()
                new_hash = calculate_dhash(image_bytes)
                if new_hash is not None:
                    audit_submission(new_hash, existing_records)
            else:
                st.info("Click **Run Image Audit** to check for near-duplicates.")
    else:
        st.info("Please upload an image to begin the audit.")

# ==================== SESSION STATE ====================
if "user" not in st.session_state:
    st.session_state.user = None
if "show_login" not in st.session_state:
    st.session_state.show_login = False
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False
if "show_ai_audit" not in st.session_state:
    st.session_state.show_ai_audit = False

# ==================== DIALOGS (POPUPS) ====================
has_dialog = hasattr(st, "dialog")

if has_dialog:
    @st.dialog("🔐 Login")
    def login_dialog():
        with st.form("login_form", clear_on_submit=False):
            u = st.text_input("Username", placeholder="jane_doe")
            p = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Login")
        colx, coly = st.columns(2)
        if colx.button("Close", use_container_width=True):
            st.session_state.show_login = False
            st.rerun()
        if submitted:
            user = verify_user(u.strip(), p)
            if user:
                st.session_state.user = user
                st.session_state.show_login = False
                st.toast(f"Welcome back, {user[1]}! 🎉", icon="✅")
                st.rerun()
            else:
                st.error("Invalid credentials.")

    @st.dialog("🆕 Sign Up")
    def signup_dialog():
        with st.form("signup_form"):
            u = st.text_input("Choose a username", placeholder="jane_doe")
            p = st.text_input("Choose a password", type="password", placeholder="Strong & memorable")
            role = st.selectbox("Role", ["user", "admin"])
            submitted = st.form_submit_button("Create Account")
        colx, coly = st.columns(2)
        if colx.button("Close", use_container_width=True):
            st.session_state.show_signup = False
            st.rerun()
        if submitted:
            if not u or not p:
                st.warning("Please fill all fields.")
            elif add_user(u.strip(), p, role):
                st.success("Account created! You can now log in.")
                st.session_state.show_signup = False
                st.session_state.show_login = True
                st.rerun()
            else:
                st.warning("That username already exists. Try another.")

    @st.dialog("🧠 AI Image Audit")
    def ai_audit_dialog():
        render_ai_audit_dialog()
        if st.button("Close", use_container_width=True):
            st.session_state.show_ai_audit = False
            st.rerun()
else:
    # Fallback using popovers if modals unavailable
    def login_dialog(): pass
    def signup_dialog(): pass
    def ai_audit_dialog(): pass

# ==================== ROUTING ====================
if st.session_state.user is None:
    # -------- HOME PAGE WITH POPUPS --------
    col = st.columns([1, 1, 1])[1]
    with col:
        st.markdown(
            """
            <div class="hero">
              <h1>🛰️ Disaster Aid Auditing Platform</h1>
              <div class="sub">Transparency • Traceability • Trust</div>
              <div style="margin-top:12px;">
                <span class="chip">Secure by design</span>
                <span class="chip">SQLite local</span>
                <span class="chip">Plotly dashboards</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Login", use_container_width=True):
                st.session_state.show_login = True
        with b2:
            if st.button("Sign Up", use_container_width=True):
                st.session_state.show_signup = True
        with b3:
            if st.button("🧠 AI Audit", use_container_width=True):
                st.session_state.show_ai_audit = True

        info_cols = st.columns([1,1,1])
        with info_cols[0]:
            with st.popover("About"):
                st.write("Track donors → beneficiaries with full visibility. Audit, filter, and export.")
        with info_cols[1]:
            with st.popover("How it works"):
                st.markdown("- Create an account\n- Add aid records\n- Use Dashboard to filter & export\n- Admins can bulk delete\n- Use **AI Audit** to check duplicate photos")
        with info_cols[2]:
            with st.popover("Contact"):
                st.write("support@aidplatform.org\n\nEY Disaster Relief Center, India\n\n+91-9876543210")

        # Preview metrics
        df = get_aid_records_cached()
        total = df["amount"].sum() if not df.empty else 0
        ver = int((df["status"] == "Verified").sum()) if not df.empty else 0
        pen = int((df["status"] == "Pending").sum()) if not df.empty else 0
        st.write("")
        k1, k2, k3 = st.columns(3)
        with k1: st.markdown(f'<div class="metric-card"><div class="metric-label">Total Aid</div><div class="metric-value">{human_currency(total)}</div></div>', unsafe_allow_html=True)
        with k2: st.markdown(f'<div class="metric-card"><div class="metric-label">Verified</div><div class="metric-value">{ver}</div></div>', unsafe_allow_html=True)
        with k3: st.markdown(f'<div class="metric-card"><div class="metric-label">Pending</div><div class="metric-value">{pen}</div></div>', unsafe_allow_html=True)

    # Open dialogs if toggled
    if has_dialog and st.session_state.show_login:
        login_dialog()
    if has_dialog and st.session_state.show_signup:
        signup_dialog()
    if has_dialog and st.session_state.show_ai_audit:
        ai_audit_dialog()
else:
    # -------- APP (POST-AUTH) --------
    user = st.session_state.user
    username, role = user[1], user[3]

    with st.sidebar:
        st.markdown(f"### 👋 {username}\n**Role:** `{role}`")
        menu = st.radio("Navigation", ["🏠 Home", "📊 Dashboard", "➕ Add Aid", "🗂️ Records", "🧠 AI Audit", "📞 Contact", "🛠 Admin"])
        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state.user = None
            st.rerun()

    # ==================== HOME ====================
    if menu == "🏠 Home":
        st.markdown("## 🌍 Welcome")
        st.write("Track donors → beneficiaries with full visibility. Slice data, verify, and export in a click.")
        st.markdown('<div class="chip">Tip: Jump to “📊 Dashboard” for insights</div>', unsafe_allow_html=True)

        df = get_aid_records_cached()
        total = df["amount"].sum() if not df.empty else 0
        ver = int((df["status"] == "Verified").sum()) if not df.empty else 0
        pen = int((df["status"] == "Pending").sum()) if not df.empty else 0

        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Total Aid</div><div class="metric-value">{human_currency(total)}</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">Verified</div><div class="metric-value">{ver}</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">Pending</div><div class="metric-value">{pen}</div></div>', unsafe_allow_html=True)

    # ==================== DASHBOARD ====================
    elif menu == "📊 Dashboard":
        st.markdown("## 📈 Dashboard")
        df = get_aid_records_cached()
        if df.empty:
            st.info("No aid records yet. Add some records to see insights.")
        else:
            with st.popover("Filters"):
                fl1, fl2 = st.columns(2)
                with fl1:
                    donor_filter = st.text_input("Filter by Donor")
                    status_filter = st.multiselect("Status", ["Pending", "Verified"], default=["Pending", "Verified"])
                with fl2:
                    bene_filter = st.text_input("Filter by Beneficiary")
                    min_amt, max_amt = float(df["amount"].min()), float(df["amount"].max())
                    amount_range = st.slider("Amount range", min_amt, max_amt, (min_amt, max_amt))

            fdf = df.copy()
            if 'donor_filter' in locals() and donor_filter:
                fdf = fdf[fdf["donor"].str.contains(donor_filter, case=False, na=False)]
            if 'bene_filter' in locals() and bene_filter:
                fdf = fdf[fdf["beneficiary"].str.contains(bene_filter, case=False, na=False)]
            if 'status_filter' in locals() and status_filter:
                fdf = fdf[fdf["status"].isin(status_filter)]
            if 'amount_range' in locals():
                fdf = fdf[(fdf["amount"] >= amount_range[0]) & (fdf["amount"] <= amount_range[1])]

            total = fdf["amount"].sum() if not fdf.empty else 0
            ver = int((fdf["status"] == "Verified").sum()) if not fdf.empty else 0
            pen = int((fdf["status"] == "Pending").sum()) if not fdf.empty else 0
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Total (Filtered)</div><div class="metric-value">{human_currency(total)}</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">Verified</div><div class="metric-value">{ver}</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">Pending</div><div class="metric-value">{pen}</div></div>', unsafe_allow_html=True)

            fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce")
            chart_cols = st.columns(2)

            with chart_cols[0]:
                pie = px.pie(fdf, names="status", values="amount", title="Aid by Status", hole=.35, template="plotly_dark")
                st.plotly_chart(pie, use_container_width=True)

            with chart_cols[1]:
                monthly = fdf.dropna(subset=["date"]).copy()
                if not monthly.empty:
                    monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
                    agg = monthly.groupby("month", as_index=False)["amount"].sum()
                    bar = px.bar(agg, x="month", y="amount", title="Monthly Aid (Sum)", template="plotly_dark")
                    bar.update_layout(xaxis_title="", yaxis_title="Amount")
                    st.plotly_chart(bar, use_container_width=True)
                else:
                    st.info("No valid dates to chart yet.")

            left, right = st.columns([3, 1])
            with left:
                td = fdf.groupby("donor", as_index=False)["amount"].sum().sort_values("amount", ascending=False).head(10)
                with st.popover("Top Donors (filtered)"):
                    st.dataframe(td, use_container_width=True)
            with right:
                csv = fdf.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Export filtered CSV", data=csv, file_name="aid_records_filtered.csv", mime="text/csv")

            st.markdown("#### Records (filtered)")
            st.dataframe(fdf.sort_values("date", ascending=False), use_container_width=True)

    # ==================== ADD AID ====================
    elif menu == "➕ Add Aid":
        st.markdown("## ➕ Add Aid Record")
        with st.popover("Add Record (Popup)", use_container_width=True):
            with st.form("add_form", clear_on_submit=True):
                donor = st.text_input("Donor Name", placeholder="e.g., ACME Foundation")
                beneficiary = st.text_input("Beneficiary Name", placeholder="e.g., Flood Relief Camp #12")
                amount = st.number_input("Amount Donated", min_value=0.0, step=100.0, help="Use your base currency")
                colx, coly = st.columns([1,1])
                with colx:
                    status = st.selectbox("Status", ["Pending", "Verified"])
                with coly:
                    date_input = st.date_input("Date", value=datetime.now())
                submitted = st.form_submit_button("Add Record")
            if submitted:
                if donor.strip() and beneficiary.strip():
                    add_aid_record(donor.strip(), beneficiary.strip(), amount, status, date_input.strftime("%Y-%m-%d"))
                    st.success("Aid record added successfully!")
                    st.toast("Record saved ✅", icon="💾")
                    st.rerun()
                else:
                    st.warning("Please fill all fields.")

    # ==================== RECORDS ====================
    elif menu == "🗂️ Records":
        st.markdown("## 🗂️ All Records")
        df = get_aid_records_cached()
        if df.empty:
            st.info("No records yet.")
        else:
            q = st.text_input("Search (matches donor, beneficiary, status)")
            sdf = df.copy()
            if q:
                mask = (
                    sdf["donor"].str.contains(q, case=False, na=False) |
                    sdf["beneficiary"].str.contains(q, case=False, na=False) |
                    sdf["status"].str.contains(q, case=False, na=False)
                )
                sdf = sdf[mask]
            st.caption("Tip: Use the download button in Dashboard for filtered exports.")
            st.dataframe(sdf.sort_values("date", ascending=False), use_container_width=True)

    # ==================== AI AUDIT ====================
    elif menu == "🧠 AI Audit":
        # Button to open modal
        if has_dialog:
            if st.button("Open AI Image Audit", use_container_width=True):
                st.session_state.show_ai_audit = True
                st.rerun()
            if st.session_state.show_ai_audit:
                ai_audit_dialog()
        else:
            st.warning("Upgrade Streamlit to 1.31+ for modal popups. Showing inline UI instead.")
            render_ai_audit_dialog()

    # ==================== CONTACT ====================
    elif menu == "📞 Contact":
        st.markdown("## 📞 Contact Us")
        with st.popover("Show Contact"):
            st.write("**Email:** support@aidplatform.org")
            st.write("**Address:** EY Disaster Relief Center, India")
            st.write("**Phone:** +91-9876543210")

    # ==================== ADMIN ====================
    elif menu == "🛠 Admin":
        st.markdown("## 🛠 Admin Panel")
        if role != "admin":
            st.warning("Access denied. Admins only.")
        else:
            df = get_aid_records_cached()
            if df.empty:
                st.info("No records to manage.")
            else:
                st.caption("Select rows to delete, then confirm.")
                df_show = df.sort_values("date", ascending=False).reset_index(drop=True)
                df_show["select"] = False
                edited = st.data_editor(
                    df_show,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "select": st.column_config.CheckboxColumn("Select"),
                        "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
                        "date": st.column_config.TextColumn("Date (YYYY-MM-DD)"),
                    },
                    disabled=["id", "donor", "beneficiary", "amount", "date", "status"],
                    height=420
                )
                selected_ids = edited.loc[edited["select"], "id"].tolist()
                colD1, colD2 = st.columns([1,3])
                with colD1:
                    st.write(f"Selected: **{len(selected_ids)}**")
                with colD2:
                    danger = st.toggle("Require confirmation", value=True, help="Prevents accidental deletion")
                del_btn = st.button("🗑️ Delete Selected", type="primary")
                if del_btn:
                    if danger:
                        sure = st.checkbox("I'm sure. Permanently delete the selected records.")
                        if not sure:
                            st.warning("Please confirm before deleting.")
                        else:
                            delete_records(selected_ids)
                            st.success(f"Deleted {len(selected_ids)} record(s).")
                            st.rerun()
                    else:
                        delete_records(selected_ids)
                        st.success(f"Deleted {len(selected_ids)} record(s).")
                        st.rerun()
=======
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime

# NEW: AI audit deps
import cv2
import numpy as np

# ==================== DATABASE SETUP ====================
conn = sqlite3.connect("aid_system.db", check_same_thread=False)
cur = conn.cursor()

cur.execute('''CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT
)''')

cur.execute('''CREATE TABLE IF NOT EXISTS aid_records(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    donor TEXT,
    beneficiary TEXT,
    amount REAL,
    date TEXT,
    status TEXT
)''')

conn.commit()

# ==================== PAGE SETTINGS & GLOBAL STYLE ====================
st.set_page_config(
    page_title="Aid Auditing Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.stApp { background: radial-gradient(1200px 600px at 20% 0%, #1d2733 0%, #11161b 40%, #0a0f14 100%) !important; }
h1, h2, h3, .main-title { color: #e6f1ff; letter-spacing: .2px; }
/* Hero */
.hero { border-radius: 20px; padding: 28px; background: linear-gradient(145deg, rgba(15,21,32,.9), rgba(18,26,38,.9));
         border: 1px solid rgba(255,255,255,.06); box-shadow: 0 16px 36px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04); }
.hero h1 { margin: 0 0 6px 0; font-size: 1.8rem; }
.sub { color:#9fb4c7; }
/* Metric cards */
.metric-card { border-radius: 16px; padding: 18px 18px 12px; background: linear-gradient(145deg, #0f1520, #121a26);
               border: 1px solid rgba(255,255,255,.06); box-shadow: 0 10px 24px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04); }
.metric-label { color: #8ca3b8; font-size: .85rem; }
.metric-value { color: #e8f2ff; font-size: 1.6rem; font-weight: 700; }
/* Chips */
.chip { display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px;
        background:rgba(0,191,255,.12); color:#aee4ff; font-size:.85rem; border:1px solid rgba(0,191,255,.25); }
/* Tables */
.dataframe thead tr th { background: #0e141c !important; color:#adc1d1 !important; }
</style>
""", unsafe_allow_html=True)

# ==================== HELPERS ====================
def add_user(username, password, role="user"):
    try:
        cur.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
        conn.commit()
        return True
    except Exception:
        return False

def verify_user(username, password):
    cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return cur.fetchone()

@st.cache_data(ttl=10)
def get_aid_records_cached():
    return pd.read_sql("SELECT * FROM aid_records", conn)

def refresh_cache():
    get_aid_records_cached.clear()

def add_aid_record(donor, beneficiary, amount, status, date_str=None):
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    cur.execute(
        "INSERT INTO aid_records (donor, beneficiary, amount, date, status) VALUES (?, ?, ?, ?, ?)",
        (donor, beneficiary, amount, date_str, status)
    )
    conn.commit()
    refresh_cache()

def delete_records(record_ids):
    cur.executemany("DELETE FROM aid_records WHERE id=?", [(int(rid),) for rid in record_ids])
    conn.commit()
    refresh_cache()

def human_currency(x):
    try:
        return f"${x:,.2f}"
    except Exception:
        return "$0.00"

# ==================== AI IMAGE AUDIT (INTEGRATED) ====================
# Config
HAMMING_THRESHOLD = 5

def calculate_dhash(image_bytes, hash_size=8):
    """Compute dHash from raw image bytes (safe for Streamlit uploader)."""
    try:
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Invalid image data. Please upload a valid JPG/PNG.")
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        diff = resized[:, :-1] > resized[:, 1:]
        # pack bits into a 64-bit integer
        dhash = 0
        for i, bit in enumerate(diff.flatten()):
            if bit:
                dhash |= (1 << i)
        return np.uint64(dhash)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def hamming_distance(hash1, hash2):
    """Portable Hamming distance (works for numpy uint64)."""
    return int(int(hash1) ^ int(hash2)).bit_count()

@st.cache_data
def generate_mock_hashes():
    """Deterministic mock hashes simulating ledger records."""
    ORIGINAL_HASH_MOCK = np.uint64(0x40808183878F9FBF)
    UNIQUE_HASH_MOCK = np.uint64(0xFFFFFFFFFFFFFFFF)
    return [
        ("Recorded Photo #1 (Original)", ORIGINAL_HASH_MOCK),
        ("Recorded Photo #2 (Unique)", UNIQUE_HASH_MOCK),
    ]

def audit_submission(new_hash, existing_hashes):
    """Run near-duplicate check and display results."""
    st.info(f"New Submission Hash (dHash): `{hex(int(new_hash))}`")
    flagged = False
    for description, existing_hash in existing_hashes:
        distance = hamming_distance(new_hash, existing_hash)
        st.caption(f"Comparing with existing record **{description}** …")
        st.text(f"  → Hamming Distance: {distance}")
        if distance <= HAMMING_THRESHOLD:
            st.error(f"🚫 AUDIT FAILED: Near-duplicate detected (distance {distance} ≤ {HAMMING_THRESHOLD}).")
            flagged = True
            break
    if not flagged:
        st.success("✅ AUDIT SUCCESS: Photo appears unique. Ready for blockchain logging.")
    return not flagged

# A modal popup for the whole audit flow (requires Streamlit ≥ 1.31)
def render_ai_audit_dialog():
    existing_records = generate_mock_hashes()
    uploaded_file = st.file_uploader(
        "Upload Field Worker Submission Photo (JPG or PNG)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.image(uploaded_file, caption="Submitted Photo", use_column_width=True)
        with c2:
            st.subheader("Audit Report")
            if st.button("Run Image Audit", use_container_width=True):
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()
                new_hash = calculate_dhash(image_bytes)
                if new_hash is not None:
                    audit_submission(new_hash, existing_records)
            else:
                st.info("Click **Run Image Audit** to check for near-duplicates.")
    else:
        st.info("Please upload an image to begin the audit.")

# ==================== SESSION STATE ====================
if "user" not in st.session_state:
    st.session_state.user = None
if "show_login" not in st.session_state:
    st.session_state.show_login = False
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False
if "show_ai_audit" not in st.session_state:
    st.session_state.show_ai_audit = False

# ==================== DIALOGS (POPUPS) ====================
has_dialog = hasattr(st, "dialog")

if has_dialog:
    @st.dialog("🔐 Login")
    def login_dialog():
        with st.form("login_form", clear_on_submit=False):
            u = st.text_input("Username", placeholder="jane_doe")
            p = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Login")
        colx, coly = st.columns(2)
        if colx.button("Close", use_container_width=True):
            st.session_state.show_login = False
            st.rerun()
        if submitted:
            user = verify_user(u.strip(), p)
            if user:
                st.session_state.user = user
                st.session_state.show_login = False
                st.toast(f"Welcome back, {user[1]}! 🎉", icon="✅")
                st.rerun()
            else:
                st.error("Invalid credentials.")

    @st.dialog("🆕 Sign Up")
    def signup_dialog():
        with st.form("signup_form"):
            u = st.text_input("Choose a username", placeholder="jane_doe")
            p = st.text_input("Choose a password", type="password", placeholder="Strong & memorable")
            role = st.selectbox("Role", ["user", "admin"])
            submitted = st.form_submit_button("Create Account")
        colx, coly = st.columns(2)
        if colx.button("Close", use_container_width=True):
            st.session_state.show_signup = False
            st.rerun()
        if submitted:
            if not u or not p:
                st.warning("Please fill all fields.")
            elif add_user(u.strip(), p, role):
                st.success("Account created! You can now log in.")
                st.session_state.show_signup = False
                st.session_state.show_login = True
                st.rerun()
            else:
                st.warning("That username already exists. Try another.")

    @st.dialog("🧠 AI Image Audit")
    def ai_audit_dialog():
        render_ai_audit_dialog()
        if st.button("Close", use_container_width=True):
            st.session_state.show_ai_audit = False
            st.rerun()
else:
    # Fallback using popovers if modals unavailable
    def login_dialog(): pass
    def signup_dialog(): pass
    def ai_audit_dialog(): pass

# ==================== ROUTING ====================
if st.session_state.user is None:
    # -------- HOME PAGE WITH POPUPS --------
    col = st.columns([1, 1, 1])[1]
    with col:
        st.markdown(
            """
            <div class="hero">
              <h1>🛰️ Disaster Aid Auditing Platform</h1>
              <div class="sub">Transparency • Traceability • Trust</div>
              <div style="margin-top:12px;">
                <span class="chip">Secure by design</span>
                <span class="chip">SQLite local</span>
                <span class="chip">Plotly dashboards</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Login", use_container_width=True):
                st.session_state.show_login = True
        with b2:
            if st.button("Sign Up", use_container_width=True):
                st.session_state.show_signup = True
        with b3:
            if st.button("🧠 AI Audit", use_container_width=True):
                st.session_state.show_ai_audit = True

        info_cols = st.columns([1,1,1])
        with info_cols[0]:
            with st.popover("About"):
                st.write("Track donors → beneficiaries with full visibility. Audit, filter, and export.")
        with info_cols[1]:
            with st.popover("How it works"):
                st.markdown("- Create an account\n- Add aid records\n- Use Dashboard to filter & export\n- Admins can bulk delete\n- Use **AI Audit** to check duplicate photos")
        with info_cols[2]:
            with st.popover("Contact"):
                st.write("support@aidplatform.org\n\nEY Disaster Relief Center, India\n\n+91-9876543210")

        # Preview metrics
        df = get_aid_records_cached()
        total = df["amount"].sum() if not df.empty else 0
        ver = int((df["status"] == "Verified").sum()) if not df.empty else 0
        pen = int((df["status"] == "Pending").sum()) if not df.empty else 0
        st.write("")
        k1, k2, k3 = st.columns(3)
        with k1: st.markdown(f'<div class="metric-card"><div class="metric-label">Total Aid</div><div class="metric-value">{human_currency(total)}</div></div>', unsafe_allow_html=True)
        with k2: st.markdown(f'<div class="metric-card"><div class="metric-label">Verified</div><div class="metric-value">{ver}</div></div>', unsafe_allow_html=True)
        with k3: st.markdown(f'<div class="metric-card"><div class="metric-label">Pending</div><div class="metric-value">{pen}</div></div>', unsafe_allow_html=True)

    # Open dialogs if toggled
    if has_dialog and st.session_state.show_login:
        login_dialog()
    if has_dialog and st.session_state.show_signup:
        signup_dialog()
    if has_dialog and st.session_state.show_ai_audit:
        ai_audit_dialog()
else:
    # -------- APP (POST-AUTH) --------
    user = st.session_state.user
    username, role = user[1], user[3]

    with st.sidebar:
        st.markdown(f"### 👋 {username}\n**Role:** `{role}`")
        menu = st.radio("Navigation", ["🏠 Home", "📊 Dashboard", "➕ Add Aid", "🗂️ Records", "🧠 AI Audit", "📞 Contact", "🛠 Admin"])
        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state.user = None
            st.rerun()

    # ==================== HOME ====================
    if menu == "🏠 Home":
        st.markdown("## 🌍 Welcome")
        st.write("Track donors → beneficiaries with full visibility. Slice data, verify, and export in a click.")
        st.markdown('<div class="chip">Tip: Jump to “📊 Dashboard” for insights</div>', unsafe_allow_html=True)

        df = get_aid_records_cached()
        total = df["amount"].sum() if not df.empty else 0
        ver = int((df["status"] == "Verified").sum()) if not df.empty else 0
        pen = int((df["status"] == "Pending").sum()) if not df.empty else 0

        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Total Aid</div><div class="metric-value">{human_currency(total)}</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">Verified</div><div class="metric-value">{ver}</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">Pending</div><div class="metric-value">{pen}</div></div>', unsafe_allow_html=True)

    # ==================== DASHBOARD ====================
    elif menu == "📊 Dashboard":
        st.markdown("## 📈 Dashboard")
        df = get_aid_records_cached()
        if df.empty:
            st.info("No aid records yet. Add some records to see insights.")
        else:
            with st.popover("Filters"):
                fl1, fl2 = st.columns(2)
                with fl1:
                    donor_filter = st.text_input("Filter by Donor")
                    status_filter = st.multiselect("Status", ["Pending", "Verified"], default=["Pending", "Verified"])
                with fl2:
                    bene_filter = st.text_input("Filter by Beneficiary")
                    min_amt, max_amt = float(df["amount"].min()), float(df["amount"].max())
                    amount_range = st.slider("Amount range", min_amt, max_amt, (min_amt, max_amt))

            fdf = df.copy()
            if 'donor_filter' in locals() and donor_filter:
                fdf = fdf[fdf["donor"].str.contains(donor_filter, case=False, na=False)]
            if 'bene_filter' in locals() and bene_filter:
                fdf = fdf[fdf["beneficiary"].str.contains(bene_filter, case=False, na=False)]
            if 'status_filter' in locals() and status_filter:
                fdf = fdf[fdf["status"].isin(status_filter)]
            if 'amount_range' in locals():
                fdf = fdf[(fdf["amount"] >= amount_range[0]) & (fdf["amount"] <= amount_range[1])]

            total = fdf["amount"].sum() if not fdf.empty else 0
            ver = int((fdf["status"] == "Verified").sum()) if not fdf.empty else 0
            pen = int((fdf["status"] == "Pending").sum()) if not fdf.empty else 0
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Total (Filtered)</div><div class="metric-value">{human_currency(total)}</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">Verified</div><div class="metric-value">{ver}</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">Pending</div><div class="metric-value">{pen}</div></div>', unsafe_allow_html=True)

            fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce")
            chart_cols = st.columns(2)

            with chart_cols[0]:
                pie = px.pie(fdf, names="status", values="amount", title="Aid by Status", hole=.35, template="plotly_dark")
                st.plotly_chart(pie, use_container_width=True)

            with chart_cols[1]:
                monthly = fdf.dropna(subset=["date"]).copy()
                if not monthly.empty:
                    monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
                    agg = monthly.groupby("month", as_index=False)["amount"].sum()
                    bar = px.bar(agg, x="month", y="amount", title="Monthly Aid (Sum)", template="plotly_dark")
                    bar.update_layout(xaxis_title="", yaxis_title="Amount")
                    st.plotly_chart(bar, use_container_width=True)
                else:
                    st.info("No valid dates to chart yet.")

            left, right = st.columns([3, 1])
            with left:
                td = fdf.groupby("donor", as_index=False)["amount"].sum().sort_values("amount", ascending=False).head(10)
                with st.popover("Top Donors (filtered)"):
                    st.dataframe(td, use_container_width=True)
            with right:
                csv = fdf.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Export filtered CSV", data=csv, file_name="aid_records_filtered.csv", mime="text/csv")

            st.markdown("#### Records (filtered)")
            st.dataframe(fdf.sort_values("date", ascending=False), use_container_width=True)

    # ==================== ADD AID ====================
    elif menu == "➕ Add Aid":
        st.markdown("## ➕ Add Aid Record")
        with st.popover("Add Record (Popup)", use_container_width=True):
            with st.form("add_form", clear_on_submit=True):
                donor = st.text_input("Donor Name", placeholder="e.g., ACME Foundation")
                beneficiary = st.text_input("Beneficiary Name", placeholder="e.g., Flood Relief Camp #12")
                amount = st.number_input("Amount Donated", min_value=0.0, step=100.0, help="Use your base currency")
                colx, coly = st.columns([1,1])
                with colx:
                    status = st.selectbox("Status", ["Pending", "Verified"])
                with coly:
                    date_input = st.date_input("Date", value=datetime.now())
                submitted = st.form_submit_button("Add Record")
            if submitted:
                if donor.strip() and beneficiary.strip():
                    add_aid_record(donor.strip(), beneficiary.strip(), amount, status, date_input.strftime("%Y-%m-%d"))
                    st.success("Aid record added successfully!")
                    st.toast("Record saved ✅", icon="💾")
                    st.rerun()
                else:
                    st.warning("Please fill all fields.")

    # ==================== RECORDS ====================
    elif menu == "🗂️ Records":
        st.markdown("## 🗂️ All Records")
        df = get_aid_records_cached()
        if df.empty:
            st.info("No records yet.")
        else:
            q = st.text_input("Search (matches donor, beneficiary, status)")
            sdf = df.copy()
            if q:
                mask = (
                    sdf["donor"].str.contains(q, case=False, na=False) |
                    sdf["beneficiary"].str.contains(q, case=False, na=False) |
                    sdf["status"].str.contains(q, case=False, na=False)
                )
                sdf = sdf[mask]
            st.caption("Tip: Use the download button in Dashboard for filtered exports.")
            st.dataframe(sdf.sort_values("date", ascending=False), use_container_width=True)

    # ==================== AI AUDIT ====================
    elif menu == "🧠 AI Audit":
        # Button to open modal
        if has_dialog:
            if st.button("Open AI Image Audit", use_container_width=True):
                st.session_state.show_ai_audit = True
                st.rerun()
            if st.session_state.show_ai_audit:
                ai_audit_dialog()
        else:
            st.warning("Upgrade Streamlit to 1.31+ for modal popups. Showing inline UI instead.")
            render_ai_audit_dialog()

    # ==================== CONTACT ====================
    elif menu == "📞 Contact":
        st.markdown("## 📞 Contact Us")
        with st.popover("Show Contact"):
            st.write("**Email:** support@aidplatform.org")
            st.write("**Address:** EY Disaster Relief Center, India")
            st.write("**Phone:** +91-9876543210")

    # ==================== ADMIN ====================
    elif menu == "🛠 Admin":
        st.markdown("## 🛠 Admin Panel")
        if role != "admin":
            st.warning("Access denied. Admins only.")
        else:
            df = get_aid_records_cached()
            if df.empty:
                st.info("No records to manage.")
            else:
                st.caption("Select rows to delete, then confirm.")
                df_show = df.sort_values("date", ascending=False).reset_index(drop=True)
                df_show["select"] = False
                edited = st.data_editor(
                    df_show,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "select": st.column_config.CheckboxColumn("Select"),
                        "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
                        "date": st.column_config.TextColumn("Date (YYYY-MM-DD)"),
                    },
                    disabled=["id", "donor", "beneficiary", "amount", "date", "status"],
                    height=420
                )
                selected_ids = edited.loc[edited["select"], "id"].tolist()
                colD1, colD2 = st.columns([1,3])
                with colD1:
                    st.write(f"Selected: **{len(selected_ids)}**")
                with colD2:
                    danger = st.toggle("Require confirmation", value=True, help="Prevents accidental deletion")
                del_btn = st.button("🗑️ Delete Selected", type="primary")
                if del_btn:
                    if danger:
                        sure = st.checkbox("I'm sure. Permanently delete the selected records.")
                        if not sure:
                            st.warning("Please confirm before deleting.")
                        else:
                            delete_records(selected_ids)
                            st.success(f"Deleted {len(selected_ids)} record(s).")
                            st.rerun()
                    else:
                        delete_records(selected_ids)
                        st.success(f"Deleted {len(selected_ids)} record(s).")
                        st.rerun()
>>>>>>> c2c3b51 (Initial push to Hackx repo)
