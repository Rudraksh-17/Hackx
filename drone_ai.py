import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime

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
    initial_sidebar_state="expanded"
)

# Global CSS (subtle glass cards, tighter spacing, prettier inputs)
st.markdown("""
<style>
/* App background */
.stApp { background: radial-gradient(1200px 600px at 20% 0%, #1d2733 0%, #11161b 40%, #0a0f14 100%) !important; }

/* Headings */
h1, h2, h3, .main-title { color: #e6f1ff; letter-spacing: .2px; }

/* Metric cards */
.metric-card {
  border-radius: 16px; padding: 18px 18px 12px;
  background: linear-gradient(145deg, #0f1520, #121a26);
  border: 1px solid rgba(255,255,255,.06);
  box-shadow: 0 10px 24px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04);
}
.metric-label { color: #8ca3b8; font-size: .85rem; }
.metric-value { color: #e8f2ff; font-size: 1.6rem; font-weight: 700; }

/* Chips */
.chip {
  display:inline-flex; align-items:center; gap:8px;
  padding:6px 10px; border-radius:999px;
  background:rgba(0,191,255,.12); color:#aee4ff; font-size:.85rem;
  border:1px solid rgba(0,191,255,.25);
}

/* Buttons */
.stButton>button, .primary-btn {
  background: linear-gradient(135deg, #00BFFF, #2b9dff);
  color:#fff !important; border:none; border-radius:12px; padding:10px 14px;
  font-weight:600; transition: all .15s ease; box-shadow: 0 8px 18px rgba(0,191,255,.18);
}
.stButton>button:hover, .primary-btn:hover { transform: translateY(-1px); box-shadow: 0 10px 24px rgba(0,191,255,.28); }

/* Inputs */
section[data-testid="stSidebar"] .stSelectbox, .stTextInput, .stNumberInput, .stDateInput {
  border-radius: 12px !important;
}

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

# ==================== LOGIN STATE ====================
# ==================== LOGIN STATE ====================
if "user" not in st.session_state:
    st.session_state.user = None
# ❌ Do NOT call st.rerun() here


# ==================== AUTH ====================
if not st.session_state.user:
    # --------- Welcome header
    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown("### 🛰️ Disaster Aid Auditing Platform")
        st.caption("Transparency • Traceability • Trust")
        st.markdown('<span class="chip">Secure by design</span>  <span class="chip">SQLite local</span>  <span class="chip">Plotly dashboards</span>', unsafe_allow_html=True)
    with colB:
        st.image("https://cdn-icons-png.flaticon.com/512/2922/2922738.png", width=120)

    choice = st.sidebar.radio("Login / Sign Up", ["Login", "Sign Up"], horizontal=True)

    if choice == "Login":
        st.markdown("#### 🔐 Sign in")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="jane_doe")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            show_pw = st.checkbox("Show password", value=False)
            if show_pw:
                st.write(f"Password: `{password}`")  # harmless visual aid
            submitted = st.form_submit_button("Login")
        if submitted:
            user = verify_user(username.strip(), password)
            if user:
                st.session_state.user = user
                st.toast(f"Welcome back, {user[1]}! 🎉", icon="✅")
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
    else:
        st.markdown("#### 🆕 Create an account")
        with st.form("signup_form"):
            username = st.text_input("Choose a username", placeholder="jane_doe")
            password = st.text_input("Choose a password", type="password", placeholder="Strong & memorable")
            role = st.selectbox("Role", ["user", "admin"])
            submitted = st.form_submit_button("Sign Up")
        if submitted:
            if not username or not password:
                st.warning("Please fill all fields.")
            elif add_user(username.strip(), password, role):
                st.success("Account created! You can now log in.")
            else:
                st.warning("That username already exists. Try another.")
else:
    # ==================== APP (POST-AUTH) ====================
    user = st.session_state.user
    username, role = user[1], user[3]

    with st.sidebar:
        st.markdown(f"### 👋 {username}\n**Role:** `{role}`")
        menu = st.radio("Navigation", ["🏠 Home", "📊 Dashboard", "➕ Add Aid", "🗂️ Records", "📞 Contact", "🛠 Admin"])
        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state.user = None
            st.rerun()

    # ==================== HOME ====================
    if menu == "🏠 Home":
        st.markdown("## 🌍 Welcome to the Disaster Aid Auditing Platform")
        st.write("Track donors → beneficiaries with full visibility. Slice data, verify, and export in a click.")
        st.markdown('<div class="chip">Tip: Jump to “📊 Dashboard” for insights</div>', unsafe_allow_html=True)

        # Quick stats preview
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
            # --- Filters row
            fl1, fl2, fl3, fl4 = st.columns([1.2, 1.2, 1.2, 1])
            with fl1:
                donor_filter = st.text_input("🔎 Filter by Donor")
            with fl2:
                bene_filter = st.text_input("🔎 Filter by Beneficiary")
            with fl3:
                status_filter = st.multiselect("Status", ["Pending", "Verified"], default=["Pending", "Verified"])
            with fl4:
                min_amt, max_amt = float(df["amount"].min()), float(df["amount"].max())
                amount_range = st.slider("Amount range", min_amt, max_amt, (min_amt, max_amt))

            # --- Apply filters
            fdf = df.copy()
            if donor_filter:
                fdf = fdf[fdf["donor"].str.contains(donor_filter, case=False, na=False)]
            if bene_filter:
                fdf = fdf[fdf["beneficiary"].str.contains(bene_filter, case=False, na=False)]
            if status_filter:
                fdf = fdf[fdf["status"].isin(status_filter)]
            fdf = fdf[(fdf["amount"] >= amount_range[0]) & (fdf["amount"] <= amount_range[1])]

            # --- KPIs
            total = fdf["amount"].sum() if not fdf.empty else 0
            ver = int((fdf["status"] == "Verified").sum()) if not fdf.empty else 0
            pen = int((fdf["status"] == "Pending").sum()) if not fdf.empty else 0
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Total (Filtered)</div><div class="metric-value">{human_currency(total)}</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">Verified</div><div class="metric-value">{ver}</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">Pending</div><div class="metric-value">{pen}</div></div>', unsafe_allow_html=True)

            # --- Charts
            fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce")
            chart_cols = st.columns(2)

            with chart_cols[0]:
                pie = px.pie(
                    fdf, names="status", values="amount",
                    title="Aid by Status",
                    hole=.35, template="plotly_dark"
                )
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

            # --- Top donors table + export
            left, right = st.columns([3, 1])
            with left:
                td = fdf.groupby("donor", as_index=False)["amount"].sum().sort_values("amount", ascending=False).head(10)
                st.caption("Top Donors (filtered)")
                st.dataframe(td, use_container_width=True)
            with right:
                csv = fdf.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Export filtered CSV", data=csv, file_name="aid_records_filtered.csv", mime="text/csv")

            # --- Raw table
            st.markdown("#### Records (filtered)")
            st.dataframe(fdf.sort_values("date", ascending=False), use_container_width=True)

    # ==================== ADD AID ====================
    elif menu == "➕ Add Aid":
        st.markdown("## ➕ Add Aid Record")
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
            else:
                st.warning("Please fill all fields.")

    # ==================== RECORDS (USER-FRIENDLY CRUD) ====================
    elif menu == "🗂️ Records":
        st.markdown("## 🗂️ All Records")
        df = get_aid_records_cached()
        if df.empty:
            st.info("No records yet.")
        else:
            # Lightweight search
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

    # ==================== CONTACT ====================
    elif menu == "📞 Contact":
        st.markdown("## 📞 Contact Us")
        st.write("""
**Email:** support@aidplatform.org  
**Address:** EY Disaster Relief Center, India  
**Phone:** +91-9876543210
        """)

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
                # Present a selectable table
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
                    disabled=["id", "donor", "beneficiary", "amount", "date", "status"], # read-only here
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
