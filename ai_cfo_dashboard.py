# ai_cfo_dashboard.py
# Streamlit dashboard for AI CFO-style metrics from an Excel file.
# Expects a sheet named "transaction" with columns:
# Date | Type | Category | Description | Amount

import base64
import datetime as dt
import io
import json
import os
import re
import urllib.parse
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from openai import OpenAI

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


def build_cash_bridge(df: pd.DataFrame, start: dt.date, end: dt.date) -> Optional[go.Figure]:
    """Return a Waterfall showing revenue, expenses, and net change."""
    if df.empty:
        return None

    revenue = float(df.loc[df["Type"] == "REVENUE", "Amount"].abs().sum())
    expenses = float(df.loc[df["Type"] == "EXPENSE", "Amount"].abs().sum())
    if revenue == 0 and expenses == 0:
        return None

    net = revenue - expenses
    labels = ["Revenue", "Expenses", "Net change"]
    values = [revenue, -expenses, net]
    texts = [fmt_money(v) for v in values]

    fig = go.Figure(
        go.Waterfall(
            name="Cash bridge",
            orientation="v",
            measure=["relative", "relative", "total"],
            x=labels,
            y=values,
            text=texts,
            textposition="outside",
            connector={"line": {"color": "#666"}},
        )
    )
    fig.update_layout(
        title=f"Cash Bridge ({format_dayfirst_scalar(start)} → {format_dayfirst_scalar(end)})",
        showlegend=False,
        yaxis_title="Amount",
    )
    return fig


def build_budget_variance(df: pd.DataFrame, budgets: Dict[str, float]) -> Optional[pd.DataFrame]:
    """Return a DataFrame comparing actual vs budget for the supplied categories."""
    if df.empty or not budgets:
        return None

    actual = (
        df.assign(Abs=lambda d: d["Amount"].abs())
        .groupby("Category")["Abs"]
        .sum()
    )
    rows = []
    for category, budget in budgets.items():
        if budget is None:
            continue
        actual_val = float(actual.get(category, 0.0))
        rows.append(
            {
                "Category": category,
                "Actual": actual_val,
                "Budget": float(budget),
                "Variance": actual_val - float(budget),
            }
        )

    if not rows:
        return None

    result = pd.DataFrame(rows).set_index("Category")
    return result.sort_values("Variance", ascending=False)


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag category-month spikes using a simple z-score threshold."""
    if df.empty:
        return pd.DataFrame(columns=["Category", "Month", "Spend", "Z-Score"])

    monthly = (
        df.assign(Abs=lambda d: d["Amount"].abs(), Month=df["Date"].dt.to_period("M"))
        .groupby(["Category", "Month"], as_index=False)["Abs"]
        .sum()
        .rename(columns={"Abs": "Spend"})
    )
    if monthly.empty:
        return pd.DataFrame(columns=["Category", "Month", "Spend", "Z-Score"])

    monthly["Mean"] = monthly.groupby("Category")["Spend"].transform("mean")
    monthly["Std"] = monthly.groupby("Category")["Spend"].transform("std")
    monthly["Std"] = monthly["Std"].replace(0, np.nan)
    monthly["Z-Score"] = (monthly["Spend"] - monthly["Mean"]) / monthly["Std"]
    flagged = monthly[monthly["Z-Score"].abs() >= 2].copy()
    if flagged.empty:
        return pd.DataFrame(columns=["Category", "Month", "Spend", "Z-Score"])

    flagged["Month"] = format_month_period(flagged["Month"])
    flagged["Spend"] = flagged["Spend"].map(float)
    return flagged[["Category", "Month", "Spend", "Z-Score"]].sort_values(
        ["Z-Score"], ascending=False
    )


def compute_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue and expenses per day for the filtered frame."""
    if df.empty:
        return pd.DataFrame(columns=["Day", "Revenue", "Expenses", "Net"])

    daily_rev = (
        df[df["Type"] == "REVENUE"].groupby(df["Date"].dt.date)["Amount"].apply(lambda s: s.abs().sum())
    )
    daily_exp = (
        df[df["Type"] == "EXPENSE"].groupby(df["Date"].dt.date)["Amount"].apply(lambda s: s.abs().sum())
    )
    daily = pd.concat([daily_rev.rename("Revenue"), daily_exp.rename("Expenses")], axis=1).fillna(0.0)
    daily["Net"] = daily["Revenue"] - daily["Expenses"]
    daily = daily.reset_index().rename(columns={"index": "Day", "Date": "Day"})
    daily["Day"] = pd.to_datetime(daily["Day"], errors="coerce", dayfirst=True)
    return daily.dropna(subset=["Day"]).sort_values("Day")


def compute_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue and expenses per calendar month."""
    if df.empty:
        return pd.DataFrame(columns=["YearMonth", "Revenue", "Expenses"])

    monthly_rev = (
        df[df["Type"] == "REVENUE"]
        .assign(Abs=lambda d: d["Amount"].abs(), YearMonth=lambda d: d["Date"].dt.to_period("M"))
        .groupby("YearMonth", as_index=False)["Abs"].sum()
        .rename(columns={"Abs": "Revenue"})
    )
    monthly_exp = (
        df[df["Type"] == "EXPENSE"]
        .assign(Abs=lambda d: d["Amount"].abs(), YearMonth=lambda d: d["Date"].dt.to_period("M"))
        .groupby("YearMonth", as_index=False)["Abs"].sum()
        .rename(columns={"Abs": "Expenses"})
    )
    monthly = pd.merge(monthly_rev, monthly_exp, on="YearMonth", how="outer").fillna(0.0)
    return monthly.sort_values("YearMonth")


@st.cache_data(show_spinner=False)
def load_vendor_rules_csv(data: bytes) -> Optional[pd.DataFrame]:
    """Load optional vendor rules with pattern/vendor columns."""
    if not data:
        return None
    try:
        rules = pd.read_csv(io.BytesIO(data))
    except Exception:
        return None
    expected = {"pattern", "vendor"}
    if not expected.issubset({c.strip().lower() for c in rules.columns}):
        return None
    rename = {}
    for col in rules.columns:
        lower = col.strip().lower()
        if lower in expected:
            rename[col] = lower
    rules = rules.rename(columns=rename)
    rules = rules[["pattern", "vendor"]].dropna()
    return rules


def apply_vendor_rules(df: pd.DataFrame, rules: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Annotate the Description column with vendor names based on regex/substrings."""
    if df.empty or rules is None or rules.empty:
        return df

    compiled = []
    for _, row in rules.iterrows():
        pattern = str(row["pattern"]).strip()
        vendor = str(row["vendor"]).strip()
        if not pattern or not vendor:
            continue
        try:
            compiled.append((re.compile(pattern, flags=re.IGNORECASE), vendor))
        except re.error:
            compiled.append((re.compile(re.escape(pattern), flags=re.IGNORECASE), vendor))

    if not compiled:
        return df

    vendors = []
    for desc in df["Description"].astype(str):
        vendor_label = ""
        for pattern, vendor in compiled:
            if pattern.search(desc):
                vendor_label = vendor
                break
        vendors.append(vendor_label)

    df = df.copy()
    df["Vendor"] = vendors
    return df


def top_vendors_by_spend(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Return top vendors by absolute spend."""
    if df.empty or "Vendor" not in df.columns:
        return pd.DataFrame(columns=["Vendor", "Spend"])

    spend = (
        df[df["Vendor"].astype(str) != ""]
        .assign(Abs=lambda d: d["Amount"].abs())
        .groupby("Vendor", as_index=False)["Abs"].sum()
        .rename(columns={"Abs": "Spend"})
        .sort_values("Spend", ascending=False)
        .head(limit)
    )
    return spend


def _budget_widget_key(category: str) -> str:
    """Generate a stable widget key for category budgets."""
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(category)).strip("_")
    safe = safe or "category"
    return f"budget_{safe.lower()}"


def compute_13_week_forecast(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Create a naive 13-week forward forecast using trailing averages."""
    daily = compute_daily_summary(df)
    if daily.empty:
        return None

    forecast_days = 13 * 7
    history_window = min(56, len(daily))
    recent = daily.tail(history_window)
    revenue_avg = float(recent["Revenue"].mean()) if not recent.empty else 0.0
    expenses_avg = float(recent["Expenses"].mean()) if not recent.empty else 0.0

    last_day = daily["Day"].max()
    future_index = pd.date_range(last_day + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    forecast = pd.DataFrame(
        {
            "Day": future_index,
            "Revenue": revenue_avg,
            "Expenses": expenses_avg,
        }
    )
    forecast["Net"] = forecast["Revenue"] - forecast["Expenses"]

    daily = pd.concat([daily, forecast], ignore_index=True)
    daily["is_forecast"] = daily["Day"] > last_day
    return daily


def build_forecast_chart(forecast_df: pd.DataFrame) -> Optional[go.Figure]:
    """Plot historical vs forward forecast with shading."""
    if forecast_df is None or forecast_df.empty:
        return None

    fig = go.Figure()
    hist = forecast_df[~forecast_df["is_forecast"]]
    fut = forecast_df[forecast_df["is_forecast"]]

    for name, color in [("Revenue", "#2ca02c"), ("Expenses", "#d62728")] :
        fig.add_trace(
            go.Scatter(
                x=hist["Day"],
                y=hist[name],
                name=f"Historical {name}",
                mode="lines",
                line=dict(color=color),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fut["Day"],
                y=fut[name],
                name=f"Forecast {name}",
                mode="lines",
                line=dict(color=color, dash="dash"),
                showlegend=True,
            )
        )

    if not fut.empty:
        fig.add_vrect(
            x0=fut["Day"].min(),
            x1=fut["Day"].max(),
            fillcolor="#f0f4ff",
            opacity=0.2,
            line_width=0,
            annotation_text="Forecast",
            annotation_position="top left",
        )

    fig.update_layout(
        title="13-Week Cash Forecast",
        xaxis_title=None,
        yaxis_title="Amount",
        legend_title=None,
    )
    fig.update_xaxes(tickformat="%d/%m/%Y")
    return fig


def build_dcf_sensitivity(
    df: pd.DataFrame,
    projection_years: int,
    revenue_growth: float,
    expense_growth: float,
    base_wacc: float,
    terminal_growth: float,
) -> Optional[Dict[str, object]]:
    """Create a heatmap of NPV sensitivity over WACC and terminal growth ranges."""
    if df.empty:
        return None

    wacc_values = np.linspace(max(base_wacc - 0.04, 0.001), base_wacc + 0.04, 9)
    terminal_values = np.linspace(terminal_growth - 0.02, terminal_growth + 0.02, 9)

    heat = []
    for tg in terminal_values:
        row = []
        for w in wacc_values:
            scenario = compute_dcf_projection(
                df,
                wacc=max(w, 0.0001),
                projection_years=projection_years,
                revenue_growth=revenue_growth,
                expense_growth=expense_growth,
                terminal_growth=tg,
            )
            row.append(np.nan if scenario is None else float(scenario.get("npv", np.nan)))
        heat.append(row)

    heat = np.array(heat)
    if np.all(np.isnan(heat)):
        return None

    x_labels = [f"{v*100:.1f}%" for v in wacc_values]
    y_labels = [f"{v*100:.1f}%" for v in terminal_values]
    fig = px.imshow(
        heat,
        x=x_labels,
        y=y_labels,
        labels=dict(x="WACC", y="Terminal growth", color="NPV"),
        text_auto=".2f",
    )
    fig.update_layout(
        title="DCF Sensitivity (NPV)",
        xaxis_title="WACC",
        yaxis_title="Terminal growth",
        coloraxis_colorbar=dict(title="NPV"),
    )
    return {
        "figure": fig,
        "wacc_range": (wacc_values.min(), wacc_values.max()),
        "terminal_range": (terminal_values.min(), terminal_values.max()),
    }


def build_shareable_link(
    start: dt.date,
    end: dt.date,
    selected_types: Iterable[str],
    rev_target: float,
    exp_target: float,
    mode: str,
) -> str:
    """Set query params for the current filters and return a relative shareable link."""
    params = {
        "start": [pd.to_datetime(start).strftime("%Y-%m-%d")] if start else [],
        "end": [pd.to_datetime(end).strftime("%Y-%m-%d")] if end else [],
        "types": [",".join(sorted(set(selected_types)))],
        "rev_target": [f"{rev_target:.2f}"],
        "exp_target": [f"{exp_target:.2f}"],
        "mode": [mode],
    }
    st.experimental_set_query_params(**{k: v for k, v in params.items() if v})
    query = urllib.parse.urlencode({k: v[0] for k, v in params.items() if v})
    return f"?{query}" if query else ""

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
        val = float(x)
        if not np.isfinite(val):
            return "—"
        return f"{val:,.2f}"
    except Exception:
        return "—"

def fmt_runway(x):
    if np.isinf(x):
        return "∞"
    return f"{x:.1f} mo"

def _annualized_operating_profile(df: pd.DataFrame) -> dict:
    """Annualise current revenue and expense run-rates from the filtered data."""
    if df.empty:
        return {"months": 0, "revenue": 0.0, "expenses": 0.0}

    tmp = df.copy()
    tmp["YearMonth"] = pd.to_datetime(tmp["Date"], errors="coerce", dayfirst=True).dt.to_period("M")
    months = int(tmp["YearMonth"].nunique())
    if months <= 0:
        return {"months": 0, "revenue": 0.0, "expenses": 0.0}

    rev = tmp[tmp["Type"] == "REVENUE"]["Amount"].abs().sum()
    exp = tmp[tmp["Type"] == "EXPENSE"]["Amount"].abs().sum()

    scale = 12.0 / months
    return {
        "months": months,
        "revenue": float(rev * scale),
        "expenses": float(exp * scale),
    }


def compute_dcf_projection(
    df: pd.DataFrame,
    wacc: float,
    projection_years: int,
    revenue_growth: float,
    expense_growth: float,
    terminal_growth: float,
):
    """
    Build a simple DCF schedule using revenue/expense run-rates and growth assumptions.
    Returns a dict with schedule DataFrame, NPV, and terminal value components.
    """
    base = _annualized_operating_profile(df)
    if base["months"] == 0 or (base["revenue"] == 0 and base["expenses"] == 0):
        return None

    rows = []
    pv_sum = 0.0
    for year in range(1, projection_years + 1):
        revenue = base["revenue"] * ((1 + revenue_growth) ** year)
        expenses = base["expenses"] * ((1 + expense_growth) ** year)
        fcf = revenue - expenses
        discount_factor = (1 + wacc) ** year
        discounted_fcf = fcf / discount_factor
        pv_sum += discounted_fcf
        rows.append(
            {
                "Year": year,
                "Revenue": revenue,
                "Expenses": expenses,
                "FCF": fcf,
                "Discounted FCF": discounted_fcf,
            }
        )

    terminal_value = None
    terminal_pv = None
    warning = None
    if projection_years > 0:
        last_fcf = rows[-1]["FCF"]
        if wacc <= terminal_growth:
            warning = "Terminal growth must be lower than the WACC to compute a terminal value."
        else:
            terminal_value = last_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
            terminal_pv = terminal_value / ((1 + wacc) ** projection_years)
            pv_sum += terminal_pv

    schedule = pd.DataFrame(rows)
    return {
        "base": base,
        "schedule": schedule,
        "npv": pv_sum,
        "terminal_value": terminal_value,
        "terminal_pv": terminal_pv,
        "warning": warning,
    }
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
df_raw["__row_id__"] = np.arange(len(df_raw))


# Date filters
min_ts = pd.to_datetime(df_raw["Date"], dayfirst=True, errors="coerce").min()
max_ts = pd.to_datetime(df_raw["Date"], dayfirst=True, errors="coerce").max()
if pd.isna(min_ts) or pd.isna(max_ts):
    st.error("No valid dates found after parsing.")
    st.stop()

min_d = min_ts.date()
max_d = max_ts.date()
default_range = (min_d, max_d) if min_d <= max_d else (max_d, min_d)
query_params = st.experimental_get_query_params()
start_override = query_params.get("start", [None])[0]
end_override = query_params.get("end", [None])[0]
date_override = None
if start_override and end_override:
    try:
        parsed_start = pd.to_datetime(start_override, errors="coerce", dayfirst=True)
        parsed_end = pd.to_datetime(end_override, errors="coerce", dayfirst=True)
        if pd.notna(parsed_start) and pd.notna(parsed_end):
            date_override = (parsed_start.date(), parsed_end.date())
    except Exception:
        date_override = None

type_options = ["REVENUE", "EXPENSE"]
type_override = []
if "types" in query_params:
    raw_types: Iterable[str] = query_params.get("types", [])
    for entry in raw_types:
        for token in str(entry).split(","):
            token = token.strip().upper()
            if token in type_options and token not in type_override:
                type_override.append(token)

rev_target_default = float(query_params.get("rev_target", [0.0])[0] or 0.0)
mode_override = query_params.get("mode", [None])[0] or None
exp_target_default = float(query_params.get("exp_target", [0.0])[0] or 0.0)

st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date range",
    value=date_override or default_range,
    min_value=min_d,
    max_value=max_d,
    format="DD/MM/YYYY",
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = default_range

selected_types = st.sidebar.multiselect(
    "Type",
    options=type_options,
    default=type_override or type_options,
)

st.sidebar.header("Targets")
rev_target = st.sidebar.number_input(
    "Revenue target",
    min_value=0.0,
    value=float(rev_target_default),
    step=1000.0,
    help="Target revenue for the currently selected period.",
)
exp_target = st.sidebar.number_input(
    "Expenses target",
    min_value=0.0,
    value=float(exp_target_default),
    step=1000.0,
    help="Target expenses for the currently selected period.",
)

st.sidebar.header("DCF Model Inputs")
projection_years = st.sidebar.slider(
    "Projection years",
    min_value=1,
    max_value=10,
    value=5,
)
rev_growth_pct = st.sidebar.number_input(
    "Revenue growth rate (% per year)",
    value=5.0,
    step=0.5,
    format="%0.1f",
)
exp_growth_pct = st.sidebar.number_input(
    "Expense growth rate (% per year)",
    value=3.0,
    step=0.5,
    format="%0.1f",
)
terminal_growth_pct = st.sidebar.number_input(
    "Terminal growth rate (%)",
    value=2.0,
    step=0.5,
    format="%0.1f",
)
wacc_pct = st.sidebar.number_input(
    "WACC (%)",
    min_value=0.1,
    value=10.0,
    step=0.1,
    format="%0.1f",
    help="Weighted Average Cost of Capital used to discount future cash flows.",
)

vendor_rules_file = st.sidebar.file_uploader(
    "Vendor rules CSV",
    type=["csv"],
    help="Optional mapping with columns pattern,vendor (substring or regex to vendor name).",
)
vendor_rules = None
if vendor_rules_file is not None:
    vendor_rules = load_vendor_rules_csv(vendor_rules_file.getvalue())
    if vendor_rules is None:
        st.sidebar.info("Could not parse vendor rules. Expected columns: pattern, vendor.")

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

if vendor_rules is not None:
    df = apply_vendor_rules(df, vendor_rules)

budget_categories = sorted(df["Category"].dropna().unique())[:20]
budgets: Dict[str, float] = {}
if budget_categories:
    st.sidebar.header("Budgets (this period)")
    for cat in budget_categories:
        widget_key = _budget_widget_key(cat)
        budgets[cat] = st.sidebar.number_input(
            f"Budget: {cat}",
            min_value=0.0,
            value=float(st.session_state.get(widget_key, 0.0)),
            step=500.0,
            key=widget_key,
        )
else:
    budgets = {}

# =========================
# KPIs
# =========================
kpi = kpis_from_filtered(df)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Cash balance", fmt_money(kpi["cash"]))
col2.metric("Burn rate (avg monthly expenses)", fmt_money(kpi["burn"]))
col3.metric("Runway (months)", fmt_runway(kpi["runway"]))

exp_delta = kpi["expenses"] - exp_target if exp_target else None
rev_delta = kpi["revenue"] - rev_target if rev_target else None

col4.metric(
    "Expenses",
    fmt_money(kpi["expenses"]),
    delta=fmt_money(exp_delta) if exp_delta is not None else None,
    delta_color="inverse"
)
col5.metric(
    "Revenue",
    fmt_money(kpi["revenue"]),
    delta=fmt_money(rev_delta) if rev_delta is not None else None,
    delta_color="normal"
)

if np.isfinite(kpi["runway"]) and kpi["runway"] < 6:
    st.toast(
        f"Runway alert: only {fmt_runway(kpi['runway'])} remaining based on current burn.",
        icon="⚠️",
    )
if exp_target > 0 and kpi["expenses"] > exp_target:
    st.toast(
        f"Expenses exceed target by {fmt_money(kpi['expenses'] - exp_target)}.",
        icon="🔥",
    )
if rev_target > 0 and kpi["revenue"] < rev_target:
    st.toast(
        f"Revenue is below target by {fmt_money(rev_target - kpi['revenue'])}.",
        icon="⚠️",
    )

budget_fig = None
budget_variance = build_budget_variance(df, budgets)
if budget_variance is not None and not budget_variance.empty:
    st.subheader("Budget vs Actuals by Category")
    budget_display = budget_variance.reset_index()
    budget_fig = px.bar(
        budget_display,
        x="Category",
        y=["Actual", "Budget"],
        barmode="group",
        title="Budget vs Actuals",
    )
    budget_fig.update_layout(legend_title=None, xaxis_title=None, yaxis_title="Amount")
    st.plotly_chart(budget_fig, use_container_width=True, theme="streamlit")
    variance_table = budget_display.assign(
        Actual=budget_display["Actual"].map(fmt_money),
        Budget=budget_display["Budget"].map(fmt_money),
        Variance=budget_display["Variance"].map(fmt_money),
    )
    st.dataframe(variance_table.sort_values("Variance", ascending=False), use_container_width=True)
else:
    budget_variance = None

anomalies_df = detect_anomalies(df)
if not anomalies_df.empty:
    st.toast(
        f"Detected {len(anomalies_df)} category-month anomalies (|z| ≥ 2).",
        icon="🚨",
    )
    st.subheader("Anomaly & Spike Detection")
    anomalies_display = anomalies_df.copy()
    anomalies_display["Spend"] = anomalies_display["Spend"].map(fmt_money)
    anomalies_display["Z-Score"] = anomalies_display["Z-Score"].map(lambda v: f"{v:.2f}")
    st.dataframe(anomalies_display, use_container_width=True)
else:
    anomalies_df = pd.DataFrame(columns=["Category", "Month", "Spend", "Z-Score"])

share_col = st.columns([1, 3])[0]
if "shareable_link" not in st.session_state:
    st.session_state.shareable_link = ""
if share_col.button("Copy Shareable Link"):
    share_link = build_shareable_link(
        start_date,
        end_date,
        selected_types,
        rev_target,
        exp_target,
        st.session_state.get("cfo_mode", mode_override or "Strategic"),
    )
    st.session_state.shareable_link = share_link
if st.session_state.shareable_link:
    st.text_input("Shareable link", st.session_state.shareable_link, key="shareable_link_display", disabled=True)

rev_growth = rev_growth_pct / 100.0
exp_growth = exp_growth_pct / 100.0
terminal_growth = terminal_growth_pct / 100.0
wacc = wacc_pct / 100.0

st.markdown("### 💰 Discounted Cash Flow (DCF) Valuation")
dcf_sensitivity_fig = None
dcf = compute_dcf_projection(
    df,
    wacc=wacc,
    projection_years=projection_years,
    revenue_growth=rev_growth,
    expense_growth=exp_growth,
    terminal_growth=terminal_growth,
)

if dcf is None:
    st.info("Not enough data to build a DCF model. Provide more revenue and expense history.")
else:
    base = dcf["base"]
    cols = st.columns(3)
    cols[0].metric("Annualised revenue", fmt_money(base["revenue"]))
    cols[1].metric("Annualised expenses", fmt_money(base["expenses"]))
    cols[2].metric("Projection horizon", f"{projection_years} years")

    if dcf["warning"]:
        st.warning(dcf["warning"])

    valuation = dcf["npv"]
    terminal_pv = dcf.get("terminal_pv") or 0.0
    st.metric("Enterprise value (DCF)", fmt_money(valuation))
    st.caption(
        f"Assumptions: WACC {wacc_pct:.1f}% • Revenue growth {rev_growth_pct:.1f}% • Expense growth {exp_growth_pct:.1f}% • Terminal growth {terminal_growth_pct:.1f}%"
    )

    schedule_display = dcf["schedule"].copy()
    for col in ["Revenue", "Expenses", "FCF", "Discounted FCF"]:
        schedule_display[col] = schedule_display[col].map(fmt_money)
    schedule_display["Year"] = schedule_display["Year"].apply(lambda y: f"Year {y}")

    st.dataframe(
        schedule_display,
        use_container_width=True,
    )

    sensitivity = build_dcf_sensitivity(
        df,
        projection_years=projection_years,
        revenue_growth=rev_growth,
        expense_growth=exp_growth,
        base_wacc=wacc,
        terminal_growth=terminal_growth,
    )
    if sensitivity:
        dcf_sensitivity_fig = sensitivity["figure"]
        st.plotly_chart(dcf_sensitivity_fig, use_container_width=True, theme="streamlit")
        st.caption(
            "Sensitivity ranges — WACC {:.1f}% to {:.1f}% • Terminal growth {:.1f}% to {:.1f}%".format(
                sensitivity["wacc_range"][0] * 100,
                sensitivity["wacc_range"][1] * 100,
                sensitivity["terminal_range"][0] * 100,
                sensitivity["terminal_range"][1] * 100,
            )
        )
    else:
        dcf_sensitivity_fig = None

    if dcf.get("terminal_value") is not None:
        st.caption(
            f"Terminal value: {fmt_money(dcf['terminal_value'])} (PV: {fmt_money(terminal_pv)})"
        )

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

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

def _agg_context(df, kpi, revenue_target=None, expense_target=None):
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

    # Target tracking
    targets = None
    if revenue_target or expense_target:
        targets = {
            "revenue_target": float(revenue_target) if revenue_target not in (None, 0) else None,
            "expense_target": float(expense_target) if expense_target not in (None, 0) else None,
        }
        if targets["revenue_target"] is not None:
            targets["revenue_delta"] = float(kpi["revenue"] - targets["revenue_target"])
        if targets["expense_target"] is not None:
            targets["expense_delta"] = float(kpi["expenses"] - targets["expense_target"])

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
        "targets": targets,
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
    ctx = _agg_context(df, kpi, revenue_target=rev_target, expense_target=exp_target)
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
        "Respond with Markdown using this structure:\n"
        "1. **CEO Request Response** – Open with 2–3 sentences that address the CEO directly, cite at least two concrete metrics, and state progress versus any revenue/expense targets when provided (actual, target, variance).\n"
        "2. **CFO Insights** – Provide a bullet list of 3–4 data-driven observations covering trends, runway, category mix, or target gaps. Keep every bullet concise and quantified.\n"
        "3. **Next Steps** – Close with a numbered list of 2–3 decisive actions for the CEO.\n"
        "Avoid generic advice, keep focus on provided data, and do not mention code."
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
if "cfo_mode" not in st.session_state: st.session_state.cfo_mode = mode_override or "Strategic"

colA, colB, colC = st.columns(3)
st.session_state.cfo_tone = colA.selectbox("Tone", ["direct","supportive","challenging"], index=0)
st.session_state.cfo_length = colB.selectbox("Length", ["short","medium","long"], index=1)
mode_options = ["Strategic", "Tactical", "Urgent"]
mode_index = mode_options.index(st.session_state.cfo_mode) if st.session_state.cfo_mode in mode_options else 0
st.session_state.cfo_mode = colC.radio("Conversation type", mode_options, index=mode_index)

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
        charts_html += f'<img alt="Cash Bridge" style="max-width:100%" src="data:image/png;base64,{fig_to_b64(cash_bridge_fig)}" />'
    except Exception:
        charts_html += "<p>Cash bridge unavailable.</p>"
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


vendor_summary: Optional[pd.DataFrame] = None
if vendor_rules is not None and "Vendor" in df.columns:
    vendor_summary = top_vendors_by_spend(df)

if vendor_summary is not None and not vendor_summary.empty:
    st.subheader("Top Vendors by Spend")
    vendor_summary = vendor_summary.assign(
        **{
            "Cumulative %": (vendor_summary["Spend"].cumsum() / vendor_summary["Spend"].sum()) * 100,
            "Spend": vendor_summary["Spend"].astype(float),
        }
    )
    vendor_fig = go.Figure()
    vendor_fig.add_bar(x=vendor_summary["Vendor"], y=vendor_summary["Spend"], name="Spend")
    vendor_fig.add_trace(
        go.Scatter(
            x=vendor_summary["Vendor"],
            y=vendor_summary["Cumulative %"],
            name="Cumulative %",
            mode="lines+markers",
            yaxis="y2",
        ))
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
    col1, col2, col3, col4, col5 = st.columns(5)

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
        vendor_fig = go.Figure()
        vendor_fig.add_bar(x=vendor_summary["Vendor"], y=vendor_summary["Spend"], name="Spend")
        vendor_fig.add_trace(
            go.Scatter(
                x=vendor_summary["Vendor"],
                y=vendor_summary["Cumulative %"],
                name="Cumulative %",
                mode="lines+markers",
                yaxis="y2",
            )
        )
        vendor_fig.update_layout(
            title="Top Vendors by Spend",
            xaxis_title=None,
            yaxis_title="Amount",
            legend_title=None,
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
        )

    with col5:
        notes_download = notes_export.copy()
        if not notes_download.empty:
            notes_download["Date"] = format_dayfirst_series(notes_download["Date"])
            notes_download = notes_download[["Date", "Type", "Category", "Description", "Amount", "Note"]]
            notes_data = notes_download.to_csv(index=False).encode("utf-8")
            disable_notes = False
        else:
            notes_data = pd.DataFrame(columns=["Date", "Type", "Category", "Description", "Amount", "Note"]).to_csv(index=False).encode("utf-8")
            disable_notes = True
        st.download_button(
            "Notes (CSV)",
            data=notes_data,
            file_name="transaction_notes.csv",
            mime="text/csv",
            disabled=disable_notes
        )

    # Optional XLSX (in-memory) export of the three tables in separate sheets
    import io

    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Transactions")
        last10_csv.to_excel(writer, index=False, sheet_name="Last10")
        top5_exp_csv.to_excel(writer, index=False, sheet_name="Top5_Expenses")
        top5_rev_csv.to_excel(writer, index=False, sheet_name="Top5_Revenues")
        notes_sheet = notes_export.copy() if not notes_export.empty else pd.DataFrame(columns=["Date", "Type", "Category", "Description", "Amount", "Note"])
        if not notes_sheet.empty:
            notes_sheet = notes_sheet.copy()
            notes_sheet["Date"] = format_dayfirst_series(notes_sheet["Date"])
        notes_sheet.to_excel(writer, index=False, sheet_name="Notes")
    xlsx_buf.seek(0)
    st.download_button(
        "All tables (XLSX)",
        data=xlsx_buf.getvalue(),
        file_name="ai_cfo_tables.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    vendor_fig.update_layout(
        title="Top Vendors by Spend",
        xaxis_title=None,
        yaxis_title="Amount",
        legend_title=None,
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
    )
    st.plotly_chart(vendor_fig, use_container_width=True, theme="streamlit")
    vendor_display = vendor_summary.copy()
    vendor_display["Spend"] = vendor_display["Spend"].map(fmt_money)
    vendor_display["Cumulative %"] = vendor_display["Cumulative %"].map(lambda v: f"{v:.1f}%")
    st.dataframe(vendor_display, use_container_width=True)
elif vendor_rules_file is not None and vendor_rules is not None:
    st.info("No vendor matches found with the provided rules.")
elif vendor_rules_file is not None:
    st.info("Vendor rules supplied but no matches detected in the filtered data.")

if vendor_rules is not None and "Vendor" in df.columns:
    vendor_summary = top_vendors_by_spend(df)
    if vendor_summary is not None and not vendor_summary.empty:
        st.subheader("Top Vendors by Spend")
        vendor_summary = vendor_summary.assign(
            **{
                "Cumulative %": (vendor_summary["Spend"].cumsum() / vendor_summary["Spend"].sum()) * 100,
                "Spend": vendor_summary["Spend"].astype(float),
            }
        )
        vendor_fig = go.Figure()
        vendor_fig.add_bar(x=vendor_summary["Vendor"], y=vendor_summary["Spend"], name="Spend")
        vendor_fig.add_trace(
            go.Scatter(
                x=vendor_summary["Vendor"],
                y=vendor_summary["Cumulative %"],
                name="Cumulative %",
                mode="lines+markers",
                yaxis="y2",
            )
        )
        vendor_fig.update_layout(
            title="Top Vendors by Spend",
            xaxis_title=None,
            yaxis_title="Amount",
            legend_title=None,
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
        )
        st.plotly_chart(vendor_fig, use_container_width=True, theme="streamlit")
        vendor_display = vendor_summary.copy()
        vendor_display["Spend"] = vendor_display["Spend"].map(fmt_money)
        vendor_display["Cumulative %"] = vendor_display["Cumulative %"].map(lambda v: f"{v:.1f}%")
        st.dataframe(vendor_display, use_container_width=True)
    else:
        st.info("No vendor matches found with the provided rules.")
elif vendor_rules_file is not None:
    st.info("Vendor rules supplied but no matches detected in the filtered data.")

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

notes_section = df.sort_values("Date", ascending=False).head(200).copy()
if "transaction_notes" not in st.session_state:
    st.session_state.transaction_notes = {}
notes_map = st.session_state.transaction_notes
if notes_map:
    pinned = df[df["__row_id__"].isin(notes_map.keys())]
    if not pinned.empty:
        notes_section = pd.concat([notes_section, pinned]).drop_duplicates("__row_id__", keep="first")
        notes_section = notes_section.sort_values("Date", ascending=False)
if not notes_section.empty:
    st.subheader("Notes & Decisions Log")
    editable = notes_section[["__row_id__", "Date", "Type", "Category", "Description", "Amount"]].copy()
    editable["Date"] = editable["Date"].dt.strftime(DATE_DISPLAY_FMT)
    editable["Amount"] = editable["Amount"].map(fmt_money)
    editable["Note"] = editable["__row_id__"].map(notes_map).fillna("")
    column_config = {
        "__row_id__": st.column_config.Column("Row ID", disabled=True),
        "Date": st.column_config.Column("Date", disabled=True),
        "Type": st.column_config.Column("Type", disabled=True),
        "Category": st.column_config.Column("Category", disabled=True),
        "Description": st.column_config.Column("Description", disabled=True),
        "Amount": st.column_config.Column("Amount", disabled=True),
    }
    edited = st.data_editor(
        editable,
        key="notes_editor",
        hide_index=True,
        column_config=column_config,
    )
    if edited is not None:
        for _, row in edited.iterrows():
            note_val = row.get("Note", "").strip()
            rid = row["__row_id__"]
            if note_val:
                notes_map[rid] = note_val
            elif rid in notes_map:
                del notes_map[rid]
    notes_export = df[df["__row_id__"].isin(notes_map.keys())].copy()
    notes_export["Note"] = notes_export["__row_id__"].map(notes_map)
    notes_export = notes_export.drop(columns=["__row_id__"], errors="ignore")[["Date", "Type", "Category", "Description", "Amount", "Note"]]
else:
    notes_export = pd.DataFrame(columns=["Date", "Type", "Category", "Description", "Amount", "Note"])

# =========================
# Charts
# =========================
st.subheader("Trends")

fig = None
fig2 = None
cash_bridge_fig = None
forecast_fig = None

if not df.empty:
    cash_bridge_fig = build_cash_bridge(df, start_date, end_date)
    if cash_bridge_fig is not None:
        st.plotly_chart(cash_bridge_fig, use_container_width=True, theme="streamlit")

    daily = compute_daily_summary(df)
    monthly = compute_monthly_summary(df)

    tab1, tab2 = st.tabs(["Daily Net", "Monthly Revenue vs Expenses"])

    with tab1:
        if daily.empty:
            st.info("Not enough daily data to chart.")
        else:
            daily_sorted = daily.sort_values("Day")
            fig = px.line(
                daily_sorted,
                x="Day",
                y=["Revenue", "Expenses", "Net"],
                title="Daily Revenue / Expenses / Net",
            )
            fig.update_layout(legend_title=None, xaxis_title=None, yaxis_title="Amount")
            fig.update_xaxes(tickformat="%d/%m/%Y")

            color_lookup = {trace.name: trace.line.color for trace in fig.data}
            ordinal_days = daily_sorted["Day"].map(pd.Timestamp.toordinal).to_numpy()
            avg_positions = {"Revenue": "top left", "Expenses": "top right", "Net": "bottom left"}

            for metric in ["Revenue", "Expenses", "Net"]:
                if metric not in daily_sorted.columns or len(daily_sorted) < 2:
                    continue
                y_vals = daily_sorted[metric].to_numpy(dtype=float)
                if not np.isfinite(y_vals).any() or np.unique(ordinal_days).size < 2:
                    continue
                coeffs = np.polyfit(ordinal_days, y_vals, 1)
                trend = coeffs[0] * ordinal_days + coeffs[1]
                fig.add_trace(
                    go.Scatter(
                        x=daily_sorted["Day"],
                        y=trend,
                        name=f"{metric} trend",
                        mode="lines",
                        line=dict(color=color_lookup.get(metric), dash="dash"),
                        showlegend=True,
                    )
                )
                avg_val = float(np.nanmean(y_vals))
                if np.isfinite(avg_val):
                    fig.add_hline(
                        y=avg_val,
                        line_dash="dot",
                        line_color=color_lookup.get(metric),
                        annotation_text=f"{metric} avg {fmt_money(avg_val)}",
                        annotation_position=avg_positions.get(metric, "top left"),
                    )

            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            st.caption(
                "Daily averages — Revenue: {} • Expenses: {} • Net: {}".format(
                    fmt_money(np.nanmean(daily_sorted["Revenue"])),
                    fmt_money(np.nanmean(daily_sorted["Expenses"])),
                    fmt_money(np.nanmean(daily_sorted["Net"])),
                )
            )

    with tab2:
        if monthly.empty:
            st.info("Not enough monthly data to chart.")
        else:
            m_df = monthly.copy()
            m_df_display = m_df.copy()
            m_df_display["YearMonthLabel"] = format_month_period(m_df_display["YearMonth"])

            fig2 = px.bar(
                m_df_display,
                x="YearMonthLabel",
                y=["Revenue", "Expenses"],
                barmode="group",
                title="Monthly Revenue vs Expenses",
            )
            fig2.update_layout(legend_title=None, xaxis_title=None, yaxis_title="Amount")
            fig2.update_xaxes(tickformat="%m/%Y")

            color_lookup2 = {trace.name: getattr(trace.marker, "color", None) for trace in fig2.data}
            month_ordinals = m_df["YearMonth"].dt.to_timestamp().map(pd.Timestamp.toordinal).to_numpy()
            avg_positions2 = {"Revenue": "top left", "Expenses": "top right"}

            for metric in ["Revenue", "Expenses"]:
                if metric not in m_df.columns or len(m_df) < 2:
                    continue
                y_vals = m_df[metric].to_numpy(dtype=float)
                if not np.isfinite(y_vals).any() or np.unique(month_ordinals).size < 2:
                    continue
                coeffs = np.polyfit(month_ordinals, y_vals, 1)
                trend = coeffs[0] * month_ordinals + coeffs[1]
                fig2.add_trace(
                    go.Scatter(
                        x=m_df_display["YearMonthLabel"],
                        y=trend,
                        name=f"{metric} trend",
                        mode="lines",
                        line=dict(color=color_lookup2.get(metric), dash="dash"),
                        showlegend=True,
                    )
                )
                avg_val = float(np.nanmean(y_vals))
                if np.isfinite(avg_val):
                    fig2.add_hline(
                        y=avg_val,
                        line_dash="dot",
                        line_color=color_lookup2.get(metric),
                        annotation_text=f"{metric} avg {fmt_money(avg_val)}",
                        annotation_position=avg_positions2.get(metric, "top left"),
                    )

            if rev_target:
                fig2.add_hline(
                    y=rev_target,
                    line_dash="dash",
                    line_color=color_lookup2.get("Revenue", "#2ca02c"),
                    annotation_text=f"Revenue target {fmt_money(rev_target)}",
                    annotation_position="bottom left",
                )
            if exp_target:
                fig2.add_hline(
                    y=exp_target,
                    line_dash="dash",
                    line_color=color_lookup2.get("Expenses", "#d62728"),
                    annotation_text=f"Expenses target {fmt_money(exp_target)}",
                    annotation_position="bottom right",
                )

            st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
            st.caption(
                "Monthly averages — Revenue: {} • Expenses: {}".format(
                    fmt_money(np.nanmean(m_df["Revenue"])),
                    fmt_money(np.nanmean(m_df["Expenses"])),
                )
            )

    forecast_df = compute_13_week_forecast(df)
    forecast_fig = build_forecast_chart(forecast_df)
    if forecast_fig is not None:
        st.plotly_chart(forecast_fig, use_container_width=True, theme="streamlit")
        if forecast_df is not None:
            future = forecast_df[forecast_df["is_forecast"]]
            if not future.empty:
                avg_daily_net = float(future["Net"].mean())
                if avg_daily_net < 0 and kpi["cash"] > 0:
                    days_to_zero = kpi["cash"] / abs(avg_daily_net)
                    projected_date = future["Day"].min() + pd.to_timedelta(days_to_zero, unit="D")
                    st.caption(
                        f"Implied runway exhaustion date: {format_dayfirst_scalar(projected_date)} (assuming forecast burn)."
                    )
else:
    st.info("No rows match your filters. Adjust the date range or Type filter.")


st.markdown("### 📥 Download")
colx, coly = st.columns(2)
with colx:
    # JSON transcript download
    transcript_json = json.dumps(st.session_state.chat, indent=2).encode("utf-8")
    st.download_button(
        "Download conversation (JSON)",
        data=transcript_json,
        file_name="ai_cfo_conversation.json",
        mime="application/json",
    )
with coly:
    # Full HTML report (KPIs + charts + transcript)
    report_bytes = build_report_html()
    st.download_button(
        "Download dashboard + conversation (HTML)",
        data=report_bytes,
        file_name="ai_cfo_report.html",
        mime="text/html",
    )
    st.markdown("### ⬇️ Data exports")
    col1, col2, col3, col4, col5 = st.columns(5)

    # Current filtered transactions → CSV
    with col1:
        tx_export = df.copy()
        tx_export["Date"] = format_dayfirst_series(tx_export["Date"])
        st.download_button(
            "Transactions (CSV)",
            data=tx_export.to_csv(index=False).encode("utf-8"),
            file_name="transactions_filtered.csv",
            mime="text/csv",
        )

    # Last 10 / Top 5s → CSV
    with col2:
        last10_csv = df.sort_values("Date", ascending=False).head(10)[
            ["Date", "Type", "Category", "Description", "Amount"]
        ]
        last10_csv_export = last10_csv.copy()
        last10_csv_export["Date"] = format_dayfirst_series(last10_csv_export["Date"])
        st.download_button(
            "Last 10 (CSV)",
            data=last10_csv_export.to_csv(index=False).encode("utf-8"),
            file_name="last10.csv",
            mime="text/csv",
        )

    with col3:
        top5_exp_csv = (
            df[df["Type"] == "EXPENSE"]
            .assign(Abs=df["Amount"].abs())
            .nlargest(5, "Abs")
            .drop(columns="Abs")
        )[["Date", "Category", "Description", "Amount"]]
        top5_exp_csv_export = top5_exp_csv.copy()
        top5_exp_csv_export["Date"] = format_dayfirst_series(top5_exp_csv_export["Date"])
        st.download_button(
            "Top 5 expenses (CSV)",
            data=top5_exp_csv_export.to_csv(index=False).encode("utf-8"),
            file_name="top5_expenses.csv",
            mime="text/csv",
        )

    with col4:
        top5_rev_csv = (
            df[df["Type"] == "REVENUE"]
            .assign(Abs=df["Amount"].abs())
            .nlargest(5, "Abs")
            .drop(columns="Abs")
        )[["Date", "Category", "Description", "Amount"]]
        top5_rev_csv_export = top5_rev_csv.copy()
        top5_rev_csv_export["Date"] = format_dayfirst_series(top5_rev_csv_export["Date"])
        st.download_button(
            "Top 5 revenues (CSV)",
            data=top5_rev_csv_export.to_csv(index=False).encode("utf-8"),
            file_name="top5_revenues.csv",
            mime="text/csv",
        )

    with col5:
        notes_download = notes_export.copy()
        if not notes_download.empty:
            notes_download["Date"] = format_dayfirst_series(notes_download["Date"])
            notes_download = notes_download[
                ["Date", "Type", "Category", "Description", "Amount", "Note"]
            ]
            notes_data = notes_download.to_csv(index=False).encode("utf-8")
            disable_notes = False
        else:
            notes_data = (
                pd.DataFrame(
                    columns=["Date", "Type", "Category", "Description", "Amount", "Note"]
                )
                .to_csv(index=False)
                .encode("utf-8")
            )
            disable_notes = True
        st.download_button(
            "Notes (CSV)",
            data=notes_data,
            file_name="transaction_notes.csv",
            mime="text/csv",
            disabled=disable_notes,
        )

    # Optional XLSX (in-memory) export of the three tables in separate sheets
    import io

    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Transactions")
        last10_csv.to_excel(writer, index=False, sheet_name="Last10")
        top5_exp_csv.to_excel(writer, index=False, sheet_name="Top5_Expenses")
        top5_rev_csv.to_excel(writer, index=False, sheet_name="Top5_Revenues")
        notes_sheet = (
            notes_export.copy()
            if not notes_export.empty
            else pd.DataFrame(
                columns=["Date", "Type", "Category", "Description", "Amount", "Note"]
            )
        )
        if not notes_sheet.empty:
            notes_sheet = notes_sheet.copy()
            notes_sheet["Date"] = format_dayfirst_series(notes_sheet["Date"])
        notes_sheet.to_excel(writer, index=False, sheet_name="Notes")
    xlsx_buf.seek(0)
    st.download_button(
        "All tables (XLSX)",
        data=xlsx_buf.getvalue(),
        file_name="ai_cfo_tables.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


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


