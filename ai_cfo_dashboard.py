# ai_cfo_dashboard.py
# Streamlit dashboard for AI CFO-style metrics from an Excel file.
# Expects a sheet named "transaction" with columns:
# Date | Type | Category | Description | Amount

import datetime as dt
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AI CFO — Dashboard", layout="wide")

# =========================
# Configuration & Aliases
# =========================
REQUIRED_COLS = ["Date", "Type", "Category", "Description", "Amount"]

DATE_DISPLAY_FMT = "%d/%m/%Y"
MONTH_DISPLAY_FMT = "%m/%Y"

REV_ALIASES = {"REVENUE", "REV", "INCOME", "SALES", "SALE", "EARNINGS", "TURNOVER"}
EXP_ALIASES = {"EXPENSE", "EXP", "COST", "PURCHASE", "PURCHASES", "SPEND", "SPENDING", "OUTFLOW", "PAYMENT"}

# =========================
# Helpers
# =========================
def infer_dayfirst_preference(series: pd.Series, default: bool = True) -> bool:
    """Infer whether day-first ordering is likely for ambiguous date strings."""
    if series.empty:
        return default

    tokens = series.str.extract(r"^\s*(?P<p1>\d{1,4})-(?P<p2>\d{1,2})-(?P<p3>\d{2,4})$")
    if tokens.empty:
        return default

    p1 = pd.to_numeric(tokens["p1"], errors="coerce")
    p2 = pd.to_numeric(tokens["p2"], errors="coerce")
    mask = p1.notna() & p2.notna()
    if not mask.any():
        return default

    p1 = p1[mask]
    p2 = p2[mask]

    # Focus on plausible day/month tokens (1-31)
    plaus_mask = (p1 <= 31) & (p2 <= 31)
    if not plaus_mask.any():
        return default

    p1 = p1[plaus_mask]
    p2 = p2[plaus_mask]

    dayfirst_signal = int((p1 > 12).sum())
    monthfirst_signal = int((p2 > 12).sum())

    if monthfirst_signal > dayfirst_signal:
        return False
    if dayfirst_signal > monthfirst_signal:
        return True

    # Fully ambiguous (all <= 12) – fall back to provided default.
    return default


def smart_parse_dates(series: pd.Series) -> pd.Series:
    """
    Robustly parse a mixed 'Date' column:
      - Excel serials (1900 or 1904 epoch auto-detected)
      - EU strings (dd.mm.yyyy, dd/mm/yyyy, dd-mm-yyyy) -> auto day/month inference
      - ISO fallbacks (preserve inferred semantics)
    Returns tz-naive timestamps; unparseable -> NaT.
    """
    s = series.copy()

    # Start with NaT
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # ---------- 1) Handle numeric-looking (possible Excel serials)
    as_num = pd.to_numeric(s, errors="coerce")
    excel_mask = as_num.notna() & (as_num >= 1) & (as_num <= 120000)  # allow wider range

    if excel_mask.any():
        # Try 1900-system first
        d1 = pd.to_datetime(as_num.loc[excel_mask], unit="D",
                            origin="1899-12-30", errors="coerce")
        # If many dates land pre-1930 (or NaT-heavy), try 1904-system
        too_old = (d1.dropna() < pd.Timestamp("1930-01-01")).mean() > 0.5
        many_nat = d1.isna().mean() > 0.5
        if too_old or many_nat:
            d1 = pd.to_datetime(as_num.loc[excel_mask], unit="D",
                                origin="1904-01-01", errors="coerce")
        out.loc[excel_mask] = d1

    # ---------- 2) Preserve existing datetime/date objects as-is
    date_like_mask = s.apply(lambda x: isinstance(x, (pd.Timestamp, dt.datetime, dt.date)))
    if date_like_mask.any():
        existing = pd.to_datetime(s.loc[date_like_mask], errors="coerce")
        out.loc[date_like_mask] = existing

    # ---------- 3) Handle strings and everything else
    str_mask = ~(excel_mask | date_like_mask)
    s_str = s.loc[str_mask].astype(str).str.strip()

    # Quick normalization: replace common separators with '-' for parser stability
    s_norm = (s_str.str.replace(r"[./]", "-", regex=True)
                    .str.replace(r"\s+.*$", "", regex=True))  # drop trailing time words if any

    prefer_dayfirst = infer_dayfirst_preference(s_norm)

    # EU-first pass (auto inferred)
    parsed1 = pd.to_datetime(
        s_norm,
        errors="coerce",
        dayfirst=prefer_dayfirst,
        infer_datetime_format=True,
    )
    need2 = parsed1.isna()
    if need2.any():
        # Fallback pass maintains day-first semantics while attempting less-normalised strings
        parsed2 = pd.to_datetime(
            s_str[need2],
            errors="coerce",
            dayfirst=prefer_dayfirst,
            infer_datetime_format=True,
        )
        parsed1.loc[need2] = parsed2

    out.loc[str_mask] = parsed1

    # ---------- 4) Ensure tz-naive
    try:
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
    except Exception:
        pass

    return out


def format_dayfirst_scalar(val) -> str:
    ts = pd.to_datetime(val, errors="coerce", dayfirst=True)
    if pd.isna(ts):
        return ""
    return ts.strftime(DATE_DISPLAY_FMT)


def format_dayfirst_series(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return ts.dt.strftime(DATE_DISPLAY_FMT).fillna("")


def format_month_period(period_series: pd.Series) -> pd.Series:
    if period_series.empty:
        return period_series.astype(str)
    try:
        return period_series.dt.to_timestamp().dt.strftime(MONTH_DISPLAY_FMT).fillna("")
    except Exception:
        coerced = pd.to_datetime(period_series.astype(str), errors="coerce", dayfirst=True)
        return coerced.dt.strftime(MONTH_DISPLAY_FMT).fillna("")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        c = str(col).strip()
        for req in REQUIRED_COLS:
            if c.lower() == req.lower():
                rename_map[col] = req
                break
    return df.rename(columns=rename_map)

def normalize_type(t: str) -> str:
    t = str(t).strip().upper()
    if t in REV_ALIASES or t.startswith("REV"):
        return "REVENUE"
    if t in EXP_ALIASES or t.startswith("EXP"):
        return "EXPENSE"
    return t

def find_transaction_sheet(xls: pd.ExcelFile):
    for s in xls.sheet_names:
        if s.strip().lower() == "transaction":
            return s, True
    return xls.sheet_names[0], False

def load_excel(file: io.BytesIO) -> pd.DataFrame:
    xls = pd.ExcelFile(file)
    sheet, is_exact = find_transaction_sheet(xls)
    df = pd.read_excel(file, sheet_name=sheet, engine="openpyxl")
    df = normalize_columns(df)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found: {list(df.columns)} (Sheet: {sheet})")

    df["Date"] = smart_parse_dates(df["Date"])
    df["Type"] = df["Type"].map(normalize_type)
    df["Category"] = df["Category"].astype(str)
    df["Description"] = df["Description"].astype(str)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Date", "Amount"]).copy()
    df["__sheet_used__"] = sheet
    df["__used_exact_tx_sheet__"] = is_exact
    return df

def kpis_from_filtered(df: pd.DataFrame) -> dict:
    """Compute main metrics: revenue, expenses, cash, burn rate, runway, tables."""
    rev_mask = df["Type"] == "REVENUE"
    exp_mask = df["Type"] == "EXPENSE"

    rev_vals = df.loc[rev_mask, "Amount"].dropna().abs()
    exp_vals = df.loc[exp_mask, "Amount"].dropna().abs()

    revenue_sum = float(rev_vals.sum())
    expense_sum = float(exp_vals.sum())
    cash_balance = revenue_sum - expense_sum

    if not df.loc[exp_mask].empty:
        tmp = df.loc[exp_mask].copy()
        tmp["YearMonth"] = tmp["Date"].dt.to_period("M")
        monthly_expenses = tmp.groupby("YearMonth")["Amount"].apply(lambda s: s.abs().sum())
        burn_rate = float(monthly_expenses.mean()) if len(monthly_expenses) else 0.0
    else:
        burn_rate = 0.0

    # Runway = Cash / Burn rate
    runway = np.inf if burn_rate <= 0 else cash_balance / burn_rate

    last10 = df.sort_values("Date", ascending=False).head(10)

    top5_exp = df.loc[exp_mask].copy()
    top5_exp["__abs__"] = top5_exp["Amount"].abs()
    top5_exp = top5_exp.sort_values("__abs__", ascending=False).head(5).drop(columns="__abs__")

    top5_rev = df.loc[rev_mask].copy()
    top5_rev["__abs__"] = top5_rev["Amount"].abs()
    top5_rev = top5_rev.sort_values("__abs__", ascending=False).head(5).drop(columns="__abs__")

    return dict(
        revenue=revenue_sum,
        expenses=expense_sum,
        cash=cash_balance,
        burn=burn_rate,
        runway=runway,
        last10=last10,
        top5_exp=top5_exp,
        top5_rev=top5_rev,
    )

def fmt_money(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "—"

def fmt_runway(x):
    if np.isinf(x): return "∞"
    return f"{x:.1f} mo"

# =========================
# UI — Sidebar
# =========================
st.sidebar.header("Upload file")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx) with sheet 'transaction'", type=["xlsx"])

if not uploaded:
    st.title("AI CFO — Dashboard")
    st.info("Upload your Excel on the left. Sheet must be named **transaction** (case-insensitive).")
    st.stop()

@st.cache_data(show_spinner=False)
def _load_from_bytes(b: bytes):
    import io
    return load_excel(io.BytesIO(b))

file_bytes = uploaded.getvalue()          # <-- bytes change for each file
df_raw = _load_from_bytes(file_bytes)


# Date filters
min_ts = pd.to_datetime(df_raw["Date"], dayfirst=True, errors="coerce").min()
max_ts = pd.to_datetime(df_raw["Date"], dayfirst=True, errors="coerce").max()
if pd.isna(min_ts) or pd.isna(max_ts):
    st.error("No valid dates found after parsing.")
    st.stop()

min_d = min_ts.date()
max_d = max_ts.date()
default_range = (min_d, max_d) if min_d <= max_d else (max_d, min_d)

st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date range",
    value=default_range,
    min_value=min_d,
    max_value=max_d,
    format="DD/MM/YYYY",
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = default_range

type_options = ["REVENUE", "EXPENSE"]
selected_types = st.sidebar.multiselect("Type", options=type_options, default=type_options)

# =========================
# Apply Filters
# =========================
df = df_raw.copy()
df = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)]
if selected_types:
    df = df[df["Type"].isin(selected_types)]
# Create YearMonth once for the filtered frame
df["YearMonth"] = df["Date"].dt.to_period("M")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

# =========================
# KPIs
# =========================
kpi = kpis_from_filtered(df)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Cash balance", fmt_money(kpi["cash"]))
col2.metric("Burn rate (avg monthly expenses)", fmt_money(kpi["burn"]))
col3.metric("Runway (months)", fmt_runway(kpi["runway"]))
col4.metric("Expenses", fmt_money(kpi["expenses"]))
col5.metric("Revenue", fmt_money(kpi["revenue"]))
# --- CFO context helpers (place near other helpers, BEFORE the UI code) ---
def _periodize(df, months=1):
    if df.empty:
        return df
    end = df["Date"].max().normalize()
    start = (end - pd.DateOffset(months=months)).normalize() + pd.offsets.MonthBegin(0)
    return df[(df["Date"] >= start) & (df["Date"] <= end)]

def summarize_context(df):
    if df.empty:
        return {"empty": True}
    rev = df[df["Type"]=="REVENUE"]["Amount"].abs().sum()
    exp = df[df["Type"]=="EXPENSE"]["Amount"].abs().sum()
    cash = rev - exp

    top_exp_cat = (df[df["Type"]=="EXPENSE"]
                   .groupby("Category", dropna=False)["Amount"].sum()
                   .abs().sort_values(ascending=False).head(5).to_dict())
    top_rev_cat = (df[df["Type"]=="REVENUE"]
                   .groupby("Category", dropna=False)["Amount"].sum()
                   .abs().sort_values(ascending=False).head(5).to_dict())

    df["YearMonth"] = df["Date"].dt.to_period("M")
    mo_rev = df[df["Type"]=="REVENUE"].groupby("YearMonth")["Amount"].sum().abs()
    mo_exp = df[df["Type"]=="EXPENSE"].groupby("YearMonth")["Amount"].sum().abs()
    last_rev, prev_rev = (mo_rev.tail(2).tolist()+[0,0])[-2:]
    last_exp, prev_exp = (mo_exp.tail(2).tolist()+[0,0])[-2:]
    mom_rev = (last_rev - prev_rev) if prev_rev else None
    mom_exp = (last_exp - prev_exp) if prev_exp else None

    return {
        "empty": False,
        "revenue": float(rev),
        "expenses": float(exp),
        "cash": float(cash),
        "mom": {"rev_delta": mom_rev, "exp_delta": mom_exp},
        "top_exp_cat": top_exp_cat,
        "top_rev_cat": top_rev_cat,
        "top5_expenses": (df[df["Type"]=="EXPENSE"]
                  .assign(Abs=lambda d: d["Amount"].abs())
                  .nlargest(5, "Abs")[["Date","Category","Description","Amount"]]
                  .assign(Date=lambda d: format_dayfirst_series(d["Date"]))
                  .astype(str).values.tolist()),
        "top5_revenues": (df[df["Type"]=="REVENUE"]
                  .assign(Abs=lambda d: d["Amount"].abs())
                  .nlargest(5, "Abs")[["Date","Category","Description","Amount"]]
                  .assign(Date=lambda d: format_dayfirst_series(d["Date"]))
                  .astype(str).values.tolist()),

        "date_start": format_dayfirst_scalar(df["Date"].min()),
        "date_end": format_dayfirst_scalar(df["Date"].max()),
    }

st.markdown("---")
# --- AI CFO: tightly grounded, 100-word max, no outside info ---
from openai import OpenAI
import os, io, base64, json, datetime as dt
import plotly.io as pio
import streamlit as st
from openai import OpenAI

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
import pandas as pd
import numpy as np

def _agg_context(df, kpi):
    """
    Aggregate financial context from the transaction sheet for the AI CFO.
    Produces a compact structured dictionary for GPT input.
    """
    if df.empty:
        return {"empty": True}

    f = df.copy()
    f["YearMonth"] = pd.to_datetime(f["Date"], errors="coerce", dayfirst=True).dt.to_period("M")

    # Category-level aggregation (absolute sums)
    exp_by_cat = (
        f[f["Type"] == "EXPENSE"]
        .groupby("Category")["Amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .head(8)
        .to_dict()
    )
    rev_by_cat = (
        f[f["Type"] == "REVENUE"]
        .groupby("Category")["Amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .head(8)
        .to_dict()
    )

    # Month-over-month trends (last 12 months)
    m_rev = (
        f[f["Type"] == "REVENUE"]
        .groupby("YearMonth")["Amount"]
        .sum()
        .abs()
        .tail(12)
    )
    m_exp = (
        f[f["Type"] == "EXPENSE"]
        .groupby("YearMonth")["Amount"]
        .sum()
        .abs()
        .tail(12)
    )
    m_df = pd.DataFrame({"Revenue": m_rev, "Expenses": m_exp}).fillna(0.0)
    mom_rev = None
    mom_exp = None
    if len(m_df) >= 2:
        mom_rev = float(m_df["Revenue"].iloc[-1] - m_df["Revenue"].iloc[-2])
        mom_exp = float(m_df["Expenses"].iloc[-1] - m_df["Expenses"].iloc[-2])

    m_df_display = m_df.reset_index().rename(columns={"index": "YearMonth"})
    if "YearMonth" in m_df_display.columns:
        m_df_display["YearMonth"] = format_month_period(m_df_display["YearMonth"])

    # Top transactions
    top_exp = (
        f[f["Type"] == "EXPENSE"]
        .assign(Abs=f["Amount"].abs())
        .nlargest(5, "Abs")[["Date", "Category", "Description", "Amount"]]
        .assign(Date=lambda d: format_dayfirst_series(d["Date"]))
        .astype(str)
        .values.tolist()
    )
    top_rev = (
        f[f["Type"] == "REVENUE"]
        .assign(Abs=f["Amount"].abs())
        .nlargest(5, "Abs")[["Date", "Category", "Description", "Amount"]]
        .assign(Date=lambda d: format_dayfirst_series(d["Date"]))
        .astype(str)
        .values.tolist()
    )

    # Construct final context
    return {
        "empty": False,
        "period": {
            "start": format_dayfirst_scalar(f["Date"].min()),
            "end": format_dayfirst_scalar(f["Date"].max()),
        },
        "kpis": {
            "revenue": float(kpi["revenue"]),
            "expenses": float(kpi["expenses"]),
            "cash": float(kpi["cash"]),
            "burn_rate_monthly": float(kpi["burn"]),
            "runway_months": (
                None if np.isinf(kpi["runway"]) else float(kpi["runway"])
            ),
        },
        "mix": {
            "expense_by_category": exp_by_cat,
            "revenue_by_category": rev_by_cat,
        },
        "trend_last_12m": m_df_display.astype(str).values.tolist(),
        "mom_delta": {"revenue": mom_rev, "expenses": mom_exp},
        "top5_expenses": top_exp,
        "top5_revenues": top_rev,
    }
# --- Session state for conversation ---
if "chat" not in st.session_state:
    st.session_state.chat = []   # list of {"role":"user/assistant", "content": "...", "ts": "ISO"}
def _compact_history(turns, max_chars=1200, max_turns=8):
    """Return a compact string summary of the last N turns for grounding."""
    if not turns: return ""
    # Take last N turns
    recent = turns[-max_turns:]
    # Format: 'CEO:' for user, 'CFO:' for assistant
    lines = []
    for t in recent:
        who = "CEO" if t["role"] == "user" else "CFO"
        txt = (t["content"] or "").strip().replace("\n"," ")
        lines.append(f"{who}: {txt}")
    s = " | ".join(lines)
    return (s[:max_chars] + "…") if len(s) > max_chars else s

def ai_cfo_reply(prompt_text: str, tone="direct", length="medium", mode="Strategic") -> str:
    # Word/Token budgets
    targets = {"short":(120,350), "medium":(220,600), "long":(350,900)}
    target_words, max_tokens = targets.get(length, (220, 600))

    # Build rich financial context
    ctx = _agg_context(df, kpi)   # you already added _agg_context earlier
    if ctx.get("empty"): return "No data available. Upload transactions or widen the date range."

    # Conversation memory (last turns)
    history = _compact_history(st.session_state.chat, max_chars=1200, max_turns=8)

    # Mode-specific guidance
    mode_directives = {
        "Strategic": (
            "Focus on medium-to-long term themes: scalability, margin structure, pricing, capital planning. "
            "Tie advice to trends and category mix. Suggest sequencing and KPIs to monitor."
        ),
        "Tactical": (
            "Focus on the next 1–3 months: cost controls, vendor negotiations, payment terms, CAC payback. "
            "Provide concrete thresholds and weekly cadences."
        ),
        "Urgent": (
            "Assume cash safety is priority. Preserve runway immediately with quantified cuts or deferrals, "
            "bridge actions, and red-line alerts. Keep guidance concise and directive."
        ),
    }[mode]

    # System: firm guardrails
    system = (
        "You are a 30+ year seasoned CFO advising the CEO. "
        "STRICTLY use only the provided financial data and the CEO’s request. "
        "Do NOT discuss code, files, or implementation. Be numbers-first and executive. "
        f"Tone: {tone}. Target ~{target_words} words."
    )

    # Structure with request FIRST
    instructions = (
        "Respond in this exact order:\n"
        "CEO Request — Direct and explicit answer to the CEO’s question FIRST, citing figures from data.+ CFO Insights\n"
        "Conclusion that sumarizes next steps"
        "Avoid generic advice. No code talk."
    )

    payload = {
        "conversation_type": mode,
        "conversation_context_compact": history,
        "ceo_request": prompt_text,
        "financial_data": ctx,
        "mode_directives": mode_directives,
        "instructions": instructions,
    }

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role":"system","content": system},
            {"role":"user","content":
                f"CEO_REQUEST:\n{prompt_text}\n\n"
                f"RECENT_CONVERSATION (compressed):\n{history or 'n/a'}\n\n"
                f"FINANCIAL_DASHBOARD_DATA:\n{ctx}\n\n"
                f"MODE_DIRECTIVES:\n{mode_directives}\n\n"
                f"{instructions}"
            }
        ],
        max_output_tokens=max_tokens,
    )
    return resp.output_text.strip()

# --- Chat UI ---
# ---- Conversation controls & helpers ----
st.markdown("### 💬 Converse with your AI CFO")

# Persist user preferences
if "cfo_tone" not in st.session_state: st.session_state.cfo_tone = "direct"
if "cfo_length" not in st.session_state: st.session_state.cfo_length = "medium"
if "cfo_mode" not in st.session_state: st.session_state.cfo_mode = "Strategic"

colA, colB, colC = st.columns(3)
st.session_state.cfo_tone = colA.selectbox("Tone", ["direct","supportive","challenging"], index=0)
st.session_state.cfo_length = colB.selectbox("Length", ["short","medium","long"], index=1)
st.session_state.cfo_mode = colC.radio("Conversation type", ["Strategic","Tactical","Urgent"], index=0)

# Conversation store
if "chat" not in st.session_state: st.session_state.chat = []  # [{role, content, ts}]
if st.button("↺ Reset conversation"):
    st.session_state.chat = []
    st.rerun()

# Render history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_msg = st.chat_input("Ask your CFO…")
if user_msg:
    st.session_state.chat.append({"role":"user","content":user_msg,"ts":dt.datetime.utcnow().isoformat()})
    with st.chat_message("user"): st.write(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("CFO is reviewing your dashboard…"):
            reply = ai_cfo_reply(
                user_msg,
                tone=st.session_state.cfo_tone,
                length=st.session_state.cfo_length,
                mode=st.session_state.cfo_mode
            )
        st.write(reply)
        st.session_state.chat.append({"role":"assistant","content":reply,"ts":dt.datetime.utcnow().isoformat()})


def _moneyfmt(x):
    try: return f"{float(x):,.2f}"
    except: return str(x)

def df_to_html(df, cols, title):
    if df.empty:
        return f"<h3>{title}</h3><p>No rows.</p>"
    df2 = df.copy()
    if "Date" in df2.columns:
        df2["Date"] = format_dayfirst_series(df2["Date"])

    if "Amount" in df2.columns:
        df2["Amount"] = df2["Amount"].map(_moneyfmt)
    html = df2[cols].to_html(index=False, escape=False)
    return f"<h3>{title}</h3>{html}"

# --- Build downloadable HTML report (dashboard + chat transcript) ---
def fig_to_b64(fig) -> str:
    png = pio.to_image(fig, format="png", scale=2)  # requires kaleido
    return base64.b64encode(png).decode("utf-8")

def build_report_html() -> bytes:
    # Snap KPIs
    snap = {
        "cash": _moneyfmt(kpi['cash']),
        "burn": _moneyfmt(kpi['burn']),
        "runway": ("∞" if np.isinf(kpi["runway"]) else f"{kpi['runway']:.1f} mo"),
        "expenses": _moneyfmt(kpi['expenses']),
        "revenue": _moneyfmt(kpi['revenue']),
        "period": (
            f"{format_dayfirst_scalar(df['Date'].min())} → {format_dayfirst_scalar(df['Date'].max())}"
            if not df.empty else "n/a"
        ),
        "generated": dt.datetime.utcnow().strftime("%d/%m/%Y %H:%M UTC"),
    }

    # Charts → base64 (ignore if missing)
    charts_html = ""
    try:
        charts_html += f'<img alt="Daily Net" style="max-width:100%" src="data:image/png;base64,{fig_to_b64(fig)}" />'
    except Exception:
        charts_html += "<p>Daily chart unavailable.</p>"
    try:
        charts_html += f'<img alt="Monthly Revenue vs Expenses" style="max-width:100%;margin-top:10px" src="data:image/png;base64,{fig_to_b64(fig2)}" />'
    except Exception:
        charts_html += "<p>Monthly chart unavailable.</p>"

    # Tables (use your already-computed frames if you have them;
    # otherwise rebuild quickly from current df)
    last10 = df.sort_values("Date", ascending=False).head(80)
    top5_exp = (df[df["Type"]=="EXPENSE"]
                .assign(Abs=df["Amount"].abs())
                .nlargest(5, "Abs")
                .drop(columns="Abs"))
    top5_rev = (df[df["Type"]=="REVENUE"]
                .assign(Abs=df["Amount"].abs())
                .nlargest(5, "Abs")
                .drop(columns="Abs"))

    last10_html = df_to_html(last10, ["Date","Type","Category","Description","Amount"], "Last 10 transactions")
    top5exp_html = df_to_html(top5_exp, ["Date","Category","Description","Amount"], "Biggest 5 expenses")
    top5rev_html = df_to_html(top5_rev, ["Date","Category","Description","Amount"], "Biggest 5 revenues")

    # Transcript HTML
    transcript_items = []
    for m in st.session_state.chat:
        who = "CEO" if m["role"]=="user" else "CFO"
        t = m.get("ts","")
        body = (m["content"] or "").replace("\n","<br>")
        transcript_items.append(f"<div class='msg {m['role']}'><div class='meta'>{who} • {t}</div><div class='body'>{body}</div></div>")
    transcript_html = "\n".join(transcript_items) if transcript_items else "<p>No messages yet.</p>"

    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8" />
<title>AI CFO Report</title>
<style>
 body{{font-family:Inter,system-ui,Segoe UI,Roboto,sans-serif;margin:24px}}
 h1{{margin:0}} .muted{{opacity:.6}}
 .grid{{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:12px;margin:12px 0}}
 .card{{border:1px solid #eee;border-radius:12px;padding:12px}}
 .label{{font-size:12px;opacity:.7}} .val{{font-size:20px;font-weight:600}}
 .msg{{border:1px solid #eee;border-radius:12px;padding:10px;margin:8px 0}}
 .msg.user{{background:#fafbff}} .msg.assistant{{background:#f9fffa}}
 .meta{{font-size:12px;opacity:.6;margin-bottom:6px}}
 .body{{white-space:pre-wrap}}
 img{{border:1px solid #eee;border-radius:12px}}
 table{{border-collapse:collapse;width:100%;margin:8px 0}}
 th,td{{border:1px solid #eee;padding:8px;text-align:left}}
 th{{background:#fafafa}}
</style></head>
<body>
  <h1>AI CFO — Dashboard & Conversation</h1>
  <div class="muted">Period: {snap['period']} • Generated: {snap['generated']}</div>

  <div class="grid">
    <div class="card"><div class="label">Cash balance</div><div class="val">{snap['cash']}</div></div>
    <div class="card"><div class="label">Burn rate (mo)</div><div class="val">{snap['burn']}</div></div>
    <div class="card"><div class="label">Runway</div><div class="val">{snap['runway']}</div></div>
    <div class="card"><div class="label">Expenses</div><div class="val">{snap['expenses']}</div></div>
    <div class="card"><div class="label">Revenue</div><div class="val">{snap['revenue']}</div></div>
  </div>

  <h2>Charts</h2>
  {charts_html}

  <h2>Tables</h2>
  {last10_html}
  {top5exp_html}
  {top5rev_html}

  <h2>Conversation Transcript</h2>
  {transcript_html}
</body></html>"""
    return html.encode("utf-8")


st.markdown("### 📥 Download")
colx, coly = st.columns(2)
with colx:
    # JSON transcript download
    transcript_json = json.dumps(st.session_state.chat, indent=2).encode("utf-8")
    st.download_button("Download conversation (JSON)", data=transcript_json,
                       file_name="ai_cfo_conversation.json", mime="application/json")
with coly:
    # Full HTML report (KPIs + charts + transcript)
    report_bytes = build_report_html()
    st.download_button("Download dashboard + conversation (HTML)", data=report_bytes,
                       file_name="ai_cfo_report.html", mime="text/html")
    st.markdown("### ⬇️ Data exports")
    col1, col2, col3, col4 = st.columns(4)

    # Current filtered transactions → CSV
    with col1:
        tx_export = df.copy()
        tx_export["Date"] = format_dayfirst_series(tx_export["Date"])
        st.download_button(
            "Transactions (CSV)",
            data=tx_export.to_csv(index=False).encode("utf-8"),
            file_name="transactions_filtered.csv",
            mime="text/csv"
        )

    # Last 10 / Top 5s → CSV
    with col2:
        last10_csv = df.sort_values("Date", ascending=False).head(10)[
            ["Date", "Type", "Category", "Description", "Amount"]]
        last10_csv_export = last10_csv.copy()
        last10_csv_export["Date"] = format_dayfirst_series(last10_csv_export["Date"])
        st.download_button(
            "Last 10 (CSV)",
            data=last10_csv_export.to_csv(index=False).encode("utf-8"),
            file_name="last10.csv",
            mime="text/csv"
        )

    with col3:
        top5_exp_csv = (df[df["Type"] == "EXPENSE"]
                        .assign(Abs=df["Amount"].abs())
                        .nlargest(5, "Abs")
                        .drop(columns="Abs"))[["Date", "Category", "Description", "Amount"]]
        top5_exp_csv_export = top5_exp_csv.copy()
        top5_exp_csv_export["Date"] = format_dayfirst_series(top5_exp_csv_export["Date"])
        st.download_button(
            "Top 5 expenses (CSV)",
            data=top5_exp_csv_export.to_csv(index=False).encode("utf-8"),
            file_name="top5_expenses.csv",
            mime="text/csv"
        )

    with col4:
        top5_rev_csv = (df[df["Type"] == "REVENUE"]
                        .assign(Abs=df["Amount"].abs())
                        .nlargest(5, "Abs")
                        .drop(columns="Abs"))[["Date", "Category", "Description", "Amount"]]
        top5_rev_csv_export = top5_rev_csv.copy()
        top5_rev_csv_export["Date"] = format_dayfirst_series(top5_rev_csv_export["Date"])
        st.download_button(
            "Top 5 revenues (CSV)",
            data=top5_rev_csv_export.to_csv(index=False).encode("utf-8"),
            file_name="top5_revenues.csv",
            mime="text/csv"
        )

    # Optional XLSX (in-memory) export of the three tables in separate sheets
    import io

    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Transactions")
        last10_csv.to_excel(writer, index=False, sheet_name="Last10")
        top5_exp_csv.to_excel(writer, index=False, sheet_name="Top5_Expenses")
        top5_rev_csv.to_excel(writer, index=False, sheet_name="Top5_Revenues")
    xlsx_buf.seek(0)
    st.download_button(
        "All tables (XLSX)",
        data=xlsx_buf.getvalue(),
        file_name="ai_cfo_tables.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =========================
# Tables
# =========================
t1, t2 = st.columns([2, 1])
with t1:
    st.subheader("Last 10 transactions")
    if not kpi["last10"].empty:
        st.dataframe(
            kpi["last10"][["Date", "Type", "Category", "Description", "Amount"]]
            .assign(Date=lambda d: format_dayfirst_series(d["Date"])),
            use_container_width=True,
            height=360
        )
    else:
        st.write("No transactions in range.")
with t2:
    st.subheader("Biggest 5 expenses")
    if not kpi["top5_exp"].empty:
        st.dataframe(
            kpi["top5_exp"][["Date", "Category", "Description", "Amount"]]
            .assign(Date=lambda d: format_dayfirst_series(d["Date"])),
            use_container_width=True,
            height=180
        )
    else:
        st.write("No expenses in range.")
    st.subheader("Biggest 5 revenues")
    if not kpi["top5_rev"].empty:
        st.dataframe(
            kpi["top5_rev"][["Date", "Category", "Description", "Amount"]]
            .assign(Date=lambda d: format_dayfirst_series(d["Date"])),
            use_container_width=True,
            height=180
        )
    else:
        st.write("No revenues in range.")


# =========================
# Charts
# =========================
st.subheader("Trends")

if not df.empty:
    rev = df[df["Type"] == "REVENUE"].copy()
    exp = df[df["Type"] == "EXPENSE"].copy()

    # --- Daily aggregates (abs to avoid sign confusion)
    daily_rev = (rev.groupby(rev["Date"].dt.date)["Amount"]
               .apply(lambda s: s.abs().sum())
               .rename("Revenue"))

    daily_exp = (exp.groupby(exp["Date"].dt.date)["Amount"]
               .apply(lambda s: s.abs().sum())
               .rename("Expenses"))

    daily = pd.concat([daily_rev, daily_exp], axis=1).fillna(0.0)
    daily["Net"] = daily["Revenue"] - daily["Expenses"]
    daily = daily.reset_index().rename(columns={"index": "Day", "Date": "Day"})
    daily["Day"] = pd.to_datetime(daily["Day"], dayfirst=True)

    tab1, tab2 = st.tabs(["Daily Net", "Monthly Revenue vs Expenses"])

    with tab1:
        import plotly.express as px
        fig = px.line(
            daily.sort_values("Day"),
            x="Day", y=["Revenue", "Expenses", "Net"],
            title="Daily Revenue / Expenses / Net"
        )
        fig.update_layout(legend_title=None, xaxis_title=None, yaxis_title="Amount")
        fig.update_xaxes(tickformat="%d/%m/%Y")
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    with tab2:
        # Ensure YearMonth exists on the filtered frame
        if "YearMonth" not in df.columns:
            df["YearMonth"] = df["Date"].dt.to_period("M")

        # Build monthly aggregates using abs() and a temporary 'Abs' column
        monthly_rev = (
            rev.assign(Abs=lambda d: d["Amount"].abs())
               .groupby("YearMonth", as_index=False)["Abs"].sum()
               .rename(columns={"Abs": "Revenue"})
        )
        monthly_exp = (
            exp.assign(Abs=lambda d: d["Amount"].abs())
               .groupby("YearMonth", as_index=False)["Abs"].sum()
               .rename(columns={"Abs": "Expenses"})
        )

        # Merge and plot
        m_df = pd.merge(monthly_rev, monthly_exp, on="YearMonth", how="outer").fillna(0.0)
        m_df["YearMonth"] = format_month_period(m_df["YearMonth"])

        fig2 = px.bar(
            m_df, x="YearMonth", y=["Revenue", "Expenses"],
            barmode="group", title="Monthly Revenue vs Expenses"
        )
        fig2.update_layout(legend_title=None, xaxis_title=None, yaxis_title="Amount")
        fig2.update_xaxes(tickformat="%m/%Y")
        st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
else:
    st.info("No rows match your filters. Adjust the date range or Type filter.")


# =========================
# Audit
# =========================
with st.expander("Calculation audit"):
    left, right = st.columns(2)
    with left:
        st.write("**Distinct Type values (post-normalization)**")
        st.write(sorted(df_raw["Type"].dropna().unique()))
        st.write("**Sheet used:**", df_raw["__sheet_used__"].iloc[0])
        if not bool(df_raw["__used_exact_tx_sheet__"].iloc[0]):
            st.warning("Used the first sheet because 'transaction' wasn't found.")
        st.write("**Row counts**")
        st.write({
            "All rows": len(df_raw),
            "Filtered rows": len(df),
            "Revenues": int((df['Type'] == 'REVENUE').sum()),
            "Expenses": int((df['Type'] == 'EXPENSE').sum()),
        })

    with right:
        st.write("**Monthly expenses used for burn (filtered period)**")
        if "YearMonth" not in df.columns:
            df["YearMonth"] = df["Date"].dt.to_period("M")
        exp_f = df[df["Type"] == "EXPENSE"].copy()
        if not exp_f.empty:
            burn_series = (exp_f.assign(Abs=lambda d: d["Amount"].abs())
                           .groupby("YearMonth")["Abs"].sum())
            burn_table = burn_series.rename("Expenses").to_frame()
            burn_table.index = format_month_period(burn_table.index.to_series()).values
            st.dataframe(burn_table)
            st.write("**Burn (mean):**", fmt_money(burn_series.mean()))
        else:
            st.write("No expenses in filtered period.")

    # <- still inside "Calculation audit"
    with st.expander("Date parsing diagnostics"):
        sample = df_raw.head(30).copy()
        sample["_raw_Date"] = sample["Date"]
        parsed = smart_parse_dates(sample["_raw_Date"])
        sample["_parsed_Date"] = parsed
        sample["_parsed_Date_str"] = format_dayfirst_series(parsed)
        st.dataframe(sample[["_raw_Date", "_parsed_Date_str"]])


