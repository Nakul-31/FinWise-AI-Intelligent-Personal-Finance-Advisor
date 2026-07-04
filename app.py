"""
FinWise AI — Intelligent Personal Finance Advisor
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
import requests

# Local modules
from data_processing import (
    generate_sample_data, parse_uploaded_csv,
    build_financial_summary, get_context_string
)
from insights import (
    classify_user, generate_insights,
    detect_anomalies, generate_recommendations
)
from ml_model import (
    predict_next_month, predict_category_wise,
    plan_goal, get_prediction_context
)
from chatbot import (
    ConversationMemory, build_user_message,
    SUGGESTED_PROMPTS, parse_goal_from_query
)

# ────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FinWise AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ────────────────────────────────────────────────────────────
#  CSS LOADER
# ────────────────────────────────────────────────────────────

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ────────────────────────────────────────────────────────────
#  PLOTLY THEME
# ────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#8b9dc3", size=12),
    margin=dict(l=20, r=20, t=30, b=20),
)

# Reusable styles applied individually to avoid **kwargs key conflicts
PLOTLY_LEGEND = dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1f2d45", font=dict(color="#8b9dc3"))
PLOTLY_AXIS   = dict(gridcolor="#1f2d45", linecolor="#1f2d45", tickcolor="#8b9dc3")

COLOR_PALETTE = [
    "#4f8ef7", "#a78bfa", "#22c55e", "#f59e0b",
    "#ef4444", "#06b6d4", "#ec4899", "#84cc16",
    "#f97316", "#8b5cf6"
]


# ────────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ────────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "df": None,
        "summary": None,
        "insights": [],
        "user_profile": {},
        "recommendations": [],
        "prediction": {},
        "cat_predictions": {},
        "anomalies": pd.DataFrame(),
        "chat_history": [],
        "memory": ConversationMemory(max_turns=10),
        "data_loaded": False,
        "page": "📊 Dashboard",
        "goal_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ────────────────────────────────────────────────────────────
#  DATA PIPELINE
# ────────────────────────────────────────────────────────────

def load_and_process(df: pd.DataFrame):
    """Run the full data processing pipeline and cache results."""
    with st.spinner("Processing your financial data..."):
        st.session_state.df = df
        summary = build_financial_summary(df)
        st.session_state.summary = summary
        st.session_state.insights = generate_insights(df, summary)
        st.session_state.user_profile = classify_user(summary)
        st.session_state.recommendations = generate_recommendations(df, summary, st.session_state.user_profile)
        st.session_state.prediction = predict_next_month(df)
        st.session_state.cat_predictions = predict_category_wise(df)
        st.session_state.anomalies = detect_anomalies(df)
        st.session_state.data_loaded = True
    st.success("✅ Data processed successfully!")


# ────────────────────────────────────────────────────────────
#  ANTHROPIC API CALL
# ────────────────────────────────────────────────────────────

def call_claude(messages: list, system: str) -> str:
    """
    Call Groq API (FREE) using llama-3.3-70b-versatile.
    Priority: st.secrets → env var → sidebar input
    """
    import os
    api_key = ""

    # 1. Streamlit secrets
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass

    # 2. Environment variable
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")

    # 3. Sidebar session input
    if not api_key:
        api_key = st.session_state.get("api_key", "").strip()

    if not api_key:
        return (
            "⚠️ Paste your **Groq API key** in the 🔑 sidebar.\n\n"
            "Get one free at: https://console.groq.com/keys"
        )

    try:
        from groq import Groq
        client = Groq(api_key=api_key.strip())

        # Convert messages — Groq uses same OpenAI-style format
        # Inject system prompt as first message
        groq_messages = [{"role": "system", "content": system}] + [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=groq_messages,
            max_tokens=1000,
            temperature=0.3,
        )
        return response.choices[0].message.content

    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err.lower() or "authentication" in err.lower():
            return (
                "❌ **Invalid Groq API key.**\n\n"
                "• Make sure you copied the full key from https://console.groq.com/keys\n"
                "• Key starts with `gsk_`\n"
                "• Re-paste it in the 🔑 sidebar"
            )
        elif "rate_limit" in err.lower():
            return "⚠️ Rate limit hit. Wait 10 seconds and try again. (Groq free tier: 30 req/min)"
        return f"❌ Error: {err}"


# ────────────────────────────────────────────────────────────
#  SIDEBAR
# ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="fw-sidebar-logo">
        <h1>💼 FinWise AI</h1>
        <p>Intelligent Finance Advisor</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=[
            "📤 Upload Data",
            "📊 Dashboard",
            "🤖 Chatbot",
            "💡 Insights",
            "📈 Prediction",
            "🎯 Goals"
        ],
        index=1,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Data status
    if st.session_state.data_loaded:
        summary = st.session_state.summary
        profile = st.session_state.user_profile
        st.markdown(f"""
        <div style="padding: 12px; background: rgba(79,142,247,0.08); border-radius: 10px; border: 1px solid rgba(79,142,247,0.2);">
            <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em; color:#4a5a7a; margin-bottom:6px;">Active Dataset</div>
            <div style="font-size:0.85rem; color:#e2e8f7; font-weight:500;">{summary['months_covered']} months · {summary['total_transactions']} txns</div>
            <div style="font-size:0.8rem; color:#8b9dc3; margin-top:4px;">{profile.get('emoji','')} {profile.get('profile','')}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:12px; background:rgba(245,158,11,0.08); border-radius:10px; border:1px solid rgba(245,158,11,0.2);">
            <div style="font-size:0.82rem; color:#f59e0b;">⚠️ No data loaded<br><small style="color:#8b9dc3;">Go to Upload Data or use sample data</small></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Persistent API Key (saved to local config file) ──
    import os as _os, json as _json

    _CONFIG_FILE = _os.path.join(_os.path.dirname(__file__), ".finwise_config.json")

    def _load_saved_key():
        try:
            if _os.path.exists(_CONFIG_FILE):
                with open(_CONFIG_FILE) as _f:
                    return _json.load(_f).get("groq_api_key", "")
        except Exception:
            pass
        return ""

    def _save_key(k):
        try:
            with open(_CONFIG_FILE, "w") as _f:
                _json.dump({"groq_api_key": k}, _f)
        except Exception:
            pass

    def _delete_key():
        try:
            if _os.path.exists(_CONFIG_FILE):
                _os.remove(_CONFIG_FILE)
        except Exception:
            pass

    # Load saved key into session on first run
    if "api_key" not in st.session_state or not st.session_state["api_key"]:
        _saved = _load_saved_key()
        if _saved:
            st.session_state["api_key"] = _saved

    _current_key = st.session_state.get("api_key", "")

    if _current_key:
        # Key already saved — show status + option to change
        st.markdown(f'''<div style="padding:10px 12px;background:rgba(34,197,94,0.1);border-radius:10px;border:1px solid rgba(34,197,94,0.25);">
            <div style="font-size:0.75rem;color:#22c55e;font-weight:600;">🔑 API Key Saved</div>
            <div style="font-size:0.72rem;color:#8b9dc3;margin-top:2px;">{_current_key[:8]}...{_current_key[-4:]}</div>
        </div>''', unsafe_allow_html=True)
        if st.button("🔄 Change Key", use_container_width=True, key="change_key_btn"):
            _delete_key()
            st.session_state["api_key"] = ""
            st.rerun()
    else:
        # No key saved — show input form
        st.markdown('<div style="font-size:0.78rem;color:#f59e0b;font-weight:600;margin-bottom:6px;">🔑 Groq API Key Required</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.72rem;color:#8b9dc3;margin-bottom:8px;">Free key → <a href="https://console.groq.com/keys" target="_blank" style="color:#4f8ef7;">console.groq.com/keys</a></div>', unsafe_allow_html=True)
        _key_input = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            label_visibility="collapsed",
            key="api_key_input_field"
        )
        _col1, _col2 = st.columns(2)
        with _col1:
            if st.button("💾 Save Key", use_container_width=True, key="save_key_btn"):
                if _key_input and _key_input.strip().startswith("gsk_"):
                    _save_key(_key_input.strip())
                    st.session_state["api_key"] = _key_input.strip()
                    st.success("✅ Saved!")
                    st.rerun()
                else:
                    st.error("Key must start with gsk_")
        with _col2:
            if st.button("🔍 Test", use_container_width=True, key="test_key_btn"):
                if _key_input:
                    try:
                        from groq import Groq as _Groq
                        _Groq(api_key=_key_input.strip()).chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": "hi"}],
                            max_tokens=5
                        )
                        _save_key(_key_input.strip())
                        st.session_state["api_key"] = _key_input.strip()
                        st.success("✅ Valid & Saved!")
                        st.rerun()
                    except Exception as _e:
                        st.error("❌ Invalid key" if "401" in str(_e) or "invalid" in str(_e).lower() else f"⚠️ {_e}")

    st.markdown("---")
    if st.button("🔄 Load Sample Data", use_container_width=True):
        df = generate_sample_data(months=6)
        load_and_process(df)
        st.rerun()

    st.markdown("""
    <div style="margin-top:auto; padding-top:20px; font-size:0.7rem; color:#4a5a7a; text-align:center;">
        FinWise AI<br> Developed By Nakul Dhiman
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
#  HELPER: require data
# ────────────────────────────────────────────────────────────

def require_data():
    if not st.session_state.data_loaded:
        st.markdown("""
        <div style="text-align:center; padding:3rem; background:rgba(79,142,247,0.05); border:1px dashed #1f2d45; border-radius:18px; margin-top:2rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">📂</div>
            <h3 style="color:#e2e8f7; font-family:'Syne',sans-serif; margin-bottom:0.5rem;">No Data Loaded</h3>
            <p style="color:#8b9dc3; font-size:0.88rem;">Upload your transaction CSV or load sample data from the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
        return False
    return True


# ════════════════════════════════════════════════════════════
#  PAGE: UPLOAD DATA
# ════════════════════════════════════════════════════════════

if page == "📤 Upload Data":
    st.markdown("""
    <div class="fw-page-header">
        <h2>Upload Financial Data</h2>
        <p>Import your bank statement or transaction CSV to get started</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        uploaded = st.file_uploader(
            "Drop your CSV file here",
            type=["csv"],
            help="CSV with columns: date, amount, category, description"
        )

        if uploaded:
            try:
                df = parse_uploaded_csv(uploaded)
                st.success(f"✅ Parsed {len(df):,} transactions successfully!")
                st.dataframe(df.head(5), use_container_width=True, height=200)
                if st.button("🚀 Process & Analyze", use_container_width=True):
                    load_and_process(df)
            except Exception as e:
                st.error(f"❌ Error parsing CSV: {e}")

        st.markdown("---")
        st.markdown("**Or use sample data to explore:**")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📦 6-Month Sample", use_container_width=True):
                df = generate_sample_data(months=6)
                load_and_process(df)
                st.rerun()
        with col_b:
            if st.button("📦 12-Month Sample", use_container_width=True):
                df = generate_sample_data(months=12)
                load_and_process(df)
                st.rerun()

    with col2:
        st.markdown("""
        <div style="background:#161d2e; border:1px solid #1f2d45; border-radius:16px; padding:1.5rem;">
            <h4 style="font-family:'Syne',sans-serif; color:#e2e8f7; margin-bottom:1rem;">📋 CSV Format Guide</h4>
            <p style="color:#8b9dc3; font-size:0.82rem; margin-bottom:0.75rem;">Your CSV should have these columns:</p>
            <table style="width:100%; font-size:0.8rem; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #1f2d45;">
                    <td style="color:#4f8ef7; padding:6px 0; font-weight:600;">date</td>
                    <td style="color:#8b9dc3; padding:6px 0;">Transaction date (any format)</td>
                </tr>
                <tr style="border-bottom:1px solid #1f2d45;">
                    <td style="color:#4f8ef7; padding:6px 0; font-weight:600;">amount</td>
                    <td style="color:#8b9dc3; padding:6px 0;">Transaction amount (₹)</td>
                </tr>
                <tr style="border-bottom:1px solid #1f2d45;">
                    <td style="color:#4f8ef7; padding:6px 0; font-weight:600;">category</td>
                    <td style="color:#8b9dc3; padding:6px 0;">Expense category</td>
                </tr>
                <tr style="border-bottom:1px solid #1f2d45;">
                    <td style="color:#4f8ef7; padding:6px 0; font-weight:600;">type</td>
                    <td style="color:#8b9dc3; padding:6px 0;">'income' or 'expense'</td>
                </tr>
                <tr>
                    <td style="color:#4f8ef7; padding:6px 0; font-weight:600;">description</td>
                    <td style="color:#8b9dc3; padding:6px 0;">Transaction notes (optional)</td>
                </tr>
            </table>
            <div style="margin-top:1rem; padding:10px; background:rgba(34,197,94,0.08); border-radius:8px; border:1px solid rgba(34,197,94,0.2);">
                <p style="color:#22c55e; font-size:0.78rem; margin:0;">✅ Column names are flexible — FinWise auto-detects common formats from major banks.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════

elif page == "📊 Dashboard":
    st.markdown("""
    <div class="fw-page-header">
        <h2>Financial Dashboard</h2>
        <p>Your complete financial picture at a glance</p>
    </div>
    """, unsafe_allow_html=True)

    if not require_data():
        st.stop()

    summary = st.session_state.summary
    profile = st.session_state.user_profile
    df = st.session_state.df

    # ── User Profile Badge ──
    savings_color = "#22c55e" if profile["savings_rate"] >= 20 else ("#f59e0b" if profile["savings_rate"] >= 10 else "#ef4444")
    st.markdown(f"""
    <div class="fw-profile-badge">
        <div class="fw-profile-icon">{profile.get('emoji','💳')}</div>
        <div>
            <div class="fw-profile-name">{profile.get('profile','Unknown')} Profile</div>
            <div class="fw-profile-desc">{profile.get('description','')}</div>
            {f'<div style="margin-top:4px; font-size:0.78rem; color:#f59e0b;">{profile["trend_alert"]}</div>' if profile.get("trend_alert") else ''}
        </div>
        <div class="fw-savings-rate" style="margin-left:auto;">
            <div class="fw-savings-rate-value" style="color:{savings_color};">{profile['savings_rate']:.0f}%</div>
            <div class="fw-savings-rate-label">Savings Rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI Cards ──
    mom = summary["month_over_month_change_pct"]
    c1, c2, c3, c4 = st.columns(4)

    def kpi_card(label, value_str, change_pct=None, value_class="neutral"):
        change_html = ""
        if change_pct is not None:
            direction = "up" if change_pct > 0 else "down"
            arrow = "↑" if change_pct > 0 else "↓"
            change_html = f'<div class="fw-kpi-change {direction}">{arrow} {abs(change_pct):.1f}% MoM</div>'
        return f"""
        <div class="fw-kpi-card">
            <div class="fw-kpi-label">{label}</div>
            <div class="fw-kpi-value {value_class}">{value_str}</div>
            {change_html}
        </div>
        """

    with c1:
        st.markdown(kpi_card(
            "Total Spending",
            f"₹{summary['total_expense']:,.0f}",
            mom, "negative"
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card(
            "Monthly Average",
            f"₹{summary['avg_monthly_expense']:,.0f}",
            value_class="neutral"
        ), unsafe_allow_html=True)
    with c3:
        net = summary["net_savings"]
        st.markdown(kpi_card(
            "Net Savings",
            f"₹{net:,.0f}",
            value_class="positive" if net >= 0 else "negative"
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card(
            "Total Income",
            f"₹{summary['total_income']:,.0f}",
            value_class="neutral"
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row 1: Category Pie + Monthly Trend ──
    ch1, ch2 = st.columns([1, 1.6])

    with ch1:
        cat_data = summary["category_breakdown_last_month"]
        if cat_data:
            fig_pie = go.Figure(go.Pie(
                labels=list(cat_data.keys()),
                values=list(cat_data.values()),
                hole=0.55,
                marker=dict(
                    colors=COLOR_PALETTE,
                    line=dict(color="#0b0f1a", width=2)
                ),
                textfont=dict(size=11, color="#e2e8f7"),
                hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>"
            ))
            fig_pie.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="Category Breakdown (Last Month)", font=dict(size=13, color="#e2e8f7")),
                showlegend=True,
                legend=dict(orientation="v", x=1, y=0.5, font=dict(size=10), bgcolor="rgba(0,0,0,0)", bordercolor="#1f2d45"),
                height=320,
                annotations=[dict(
                    text=f"₹{sum(cat_data.values()):,.0f}",
                    x=0.5, y=0.5, font_size=14, showarrow=False,
                    font=dict(color="#e2e8f7", family="Syne, sans-serif")
                )]
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    with ch2:
        monthly_exp = summary["monthly_expense_trend"]
        monthly_inc = summary["monthly_income_trend"]
        months_sorted = sorted(set(list(monthly_exp.keys()) + list(monthly_inc.keys())))

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=months_sorted,
            y=[monthly_inc.get(m, 0) for m in months_sorted],
            name="Income", mode="lines+markers",
            line=dict(color="#22c55e", width=2.5),
            marker=dict(size=7, color="#22c55e"),
            fill="tozeroy", fillcolor="rgba(34,197,94,0.06)",
            hovertemplate="Income %{x}<br>₹%{y:,.0f}<extra></extra>"
        ))
        fig_trend.add_trace(go.Scatter(
            x=months_sorted,
            y=[monthly_exp.get(m, 0) for m in months_sorted],
            name="Expense", mode="lines+markers",
            line=dict(color="#ef4444", width=2.5),
            marker=dict(size=7, color="#ef4444"),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.06)",
            hovertemplate="Expense %{x}<br>₹%{y:,.0f}<extra></extra>"
        ))
        fig_trend.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Income vs Expense Trend", font=dict(size=13, color="#e2e8f7")),
            legend=PLOTLY_LEGEND,
            xaxis=PLOTLY_AXIS,
            yaxis=PLOTLY_AXIS,
            height=320,
            xaxis_tickangle=-30,
            hovermode="x unified"
        )
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

    # ── Chart Row 2: Category Bar + Weekend vs Weekday ──
    ch3, ch4 = st.columns([1.6, 1])

    with ch3:
        cat_all = summary["category_breakdown_all_time"]
        cat_sorted = dict(sorted(cat_all.items(), key=lambda x: x[1], reverse=True))
        fig_bar = go.Figure(go.Bar(
            x=list(cat_sorted.keys()),
            y=list(cat_sorted.values()),
            marker=dict(
                color=list(cat_sorted.values()),
                colorscale=[[0, "#1f2d45"], [0.5, "#4f8ef7"], [1, "#a78bfa"]],
                line=dict(color="rgba(0,0,0,0)", width=0),
                cornerradius=6
            ),
            hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>",
            text=[f"₹{v:,.0f}" for v in cat_sorted.values()],
            textposition="outside",
            textfont=dict(size=9, color="#8b9dc3")
        ))
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="All-Time Spending by Category", font=dict(size=13, color="#e2e8f7")),
            xaxis=PLOTLY_AXIS,
            yaxis=PLOTLY_AXIS,
            height=300,
            xaxis_tickangle=-30,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with ch4:
        fig_ww = go.Figure(go.Bar(
            x=["Weekday", "Weekend"],
            y=[summary["weekday_spend"], summary["weekend_spend"]],
            marker=dict(
                color=["#4f8ef7", "#a78bfa"],
                cornerradius=6
            ),
            text=[f"₹{summary['weekday_spend']:,.0f}", f"₹{summary['weekend_spend']:,.0f}"],
            textposition="outside",
            textfont=dict(size=11, color="#8b9dc3"),
            hovertemplate="%{x}<br>₹%{y:,.0f}<extra></extra>"
        ))
        fig_ww.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Weekend vs Weekday Spend", font=dict(size=13, color="#e2e8f7")),
            xaxis=PLOTLY_AXIS,
            yaxis=PLOTLY_AXIS,
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_ww, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════
#  PAGE: CHATBOT
# ════════════════════════════════════════════════════════════

elif page == "🤖 Chatbot":
    st.markdown("""
    <div class="fw-page-header">
        <h2>AI Financial Advisor</h2>
        <p>Ask anything about your finances — powered by real data analysis</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.markdown("""
        <div style="text-align:center; padding:3rem; background:rgba(79,142,247,0.05); border:1px dashed #1f2d45; border-radius:18px;">
            <div style="font-size:3rem; margin-bottom:1rem;">🤖</div>
            <h3 style="color:#e2e8f7; font-family:'Syne',sans-serif; margin-bottom:0.5rem;">Load Data First</h3>
            <p style="color:#8b9dc3; font-size:0.88rem;">FinWise AI needs your financial data to give accurate, personalized advice.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📦 Load Sample Data to Try Chatbot"):
            df = generate_sample_data(months=6)
            load_and_process(df)
            st.rerun()
        st.stop()

    # ── Chat display ──
    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="fw-empty-chat">
                <div class="fw-empty-chat-icon">🧠</div>
                <h3>FinWise AI Ready</h3>
                <p>Ask me anything about your spending, savings, predictions, or financial goals.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                role = msg["role"]
                content = msg["content"]
                time_str = msg.get("time", "")

                if role == "user":
                    st.markdown(f"""
                    <div class="fw-message user">
                        <div class="fw-message-avatar">U</div>
                        <div class="fw-message-content">
                            <div class="fw-message-bubble">{content}</div>
                            <div class="fw-message-time">{time_str}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="fw-message bot">
                        <div class="fw-message-avatar">🤖</div>
                        <div class="fw-message-content">
                            <div class="fw-message-bubble">{content}</div>
                            <div class="fw-message-time">{time_str}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Suggested prompts ──
    st.markdown("<div class='fw-prompt-chips'>", unsafe_allow_html=True)
    prompt_cols = st.columns(5)
    quick_prompts = SUGGESTED_PROMPTS[:5]
    for i, prompt in enumerate(quick_prompts):
        with prompt_cols[i]:
            if st.button(prompt, key=f"qp_{i}", use_container_width=True):
                st.session_state._pending_prompt = prompt
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Input ──
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Message",
            placeholder="Ask about your spending, savings goals, predictions...",
            key="chat_input",
            label_visibility="collapsed"
        )
    with col_send:
        send = st.button("Send →", use_container_width=True)

    # Handle pending prompt (from chip click)
    if hasattr(st.session_state, "_pending_prompt"):
        user_input = st.session_state._pending_prompt
        del st.session_state._pending_prompt
        send = True

    # ── Process message ──
    if send and user_input and user_input.strip():
        summary = st.session_state.summary
        prediction = st.session_state.prediction
        cat_preds = st.session_state.cat_predictions
        insights = st.session_state.insights
        profile = st.session_state.user_profile
        recommendations = st.session_state.recommendations
        anomalies = st.session_state.anomalies

        # Build anomaly summary
        anom_str = ""
        if not anomalies.empty:
            anom_rows = [
                f"- ₹{row['amount']:,.0f} in {row['category']} on {row['date'].strftime('%d %b')}"
                for _, row in anomalies.head(3).iterrows()
            ]
            anom_str = "Detected anomalous transactions:\n" + "\n".join(anom_rows)

        # Build contexts
        fin_ctx = get_context_string(summary)
        pred_ctx = get_prediction_context(prediction, cat_preds)

        # Check for goal query
        goal_amount, goal_months = parse_goal_from_query(user_input)
        if goal_amount and goal_months:
            goal_data = plan_goal(summary, goal_amount, goal_months, st.session_state.df)
            goal_ctx = (
                f"=== GOAL PLAN ===\n"
                f"Goal: ₹{goal_amount:,.0f} in {goal_months} months\n"
                f"Required monthly savings: ₹{goal_data['required_monthly_savings']:,.0f}\n"
                f"Current monthly savings: ₹{goal_data['current_monthly_savings']:,.0f}\n"
                f"Savings gap: ₹{goal_data['savings_gap']:,.0f}\n"
                f"Feasible: {'Yes' if goal_data['feasible'] else 'Challenging'}\n"
            )
            if goal_data["suggested_cuts"]:
                cuts = [f"- Cut {c['category']} by ₹{c['suggested_cut']:,.0f}/mo" for c in goal_data["suggested_cuts"]]
                goal_ctx += "Suggested cuts:\n" + "\n".join(cuts)
            fin_ctx = fin_ctx + "\n\n" + goal_ctx

        # Build user message with context
        full_user_msg = build_user_message(
            user_input, fin_ctx, pred_ctx, insights,
            profile, anom_str, recommendations
        )

        # Add to memory
        memory = st.session_state.memory
        memory.add("user", full_user_msg)

        # Record display message
        now = datetime.now().strftime("%I:%M %p")
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "time": now
        })

        # ── Off-topic guard (client-side, saves API calls) ──
        FINANCE_KEYWORDS = [
            "spend","spent","expense","cost","paid","pay","amount","rupee","₹",
            "income","salary","earn","saving","save","budget","goal","month",
            "category","transfer","transaction","predict","forecast","invest",
            "overspend","insight","anomaly","trend","compare","rate","balance",
            "food","shopping","transport","rent","utilities","entertainment",
            "how much","last month","this month","next month","where am i",
            "reduce","cut","recommendation","advice","profile","saver","spender"
        ]
        query_lower = user_input.lower()
        is_finance_query = any(kw in query_lower for kw in FINANCE_KEYWORDS)

        # Hard block clearly off-topic queries
        BLOCK_PATTERNS = [
            r"\bwhat is [a-z]+\b(?!.*spend|.*save|.*expense|.*income)",
            r"\bdefine\b", r"\bexplain [a-z]+ algorithm\b",
            r"\bwho is\b", r"\btell me a joke\b", r"\bwrite (a|me)\b",
            r"\bhow does .*(work|function)\b(?!.*money|.*budget|.*finance)",
            r"\b(bfs|dfs|python|java|html|css|sql|api|algorithm|sorting)\b"
        ]
        import re as _re
        is_blocked = any(_re.search(p, query_lower) for p in BLOCK_PATTERNS)

        if is_blocked or (not is_finance_query and len(user_input.split()) <= 6):
            response = "⚠️ I'm FinWise AI — I only answer questions about YOUR financial data. Try: \"Where am I overspending?\" or \"What's my savings rate?\""
            memory.add("assistant", response)
            st.session_state.chat_history.append({"role": "assistant", "content": response, "time": datetime.now().strftime("%I:%M %p")})
            st.rerun()

        # Call Claude
        with st.spinner("FinWise AI is analyzing..."):
            from chatbot import SYSTEM_PROMPT
            response = call_claude(memory.get_messages(), SYSTEM_PROMPT)

        memory.add("assistant", response)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "time": datetime.now().strftime("%I:%M %p")
        })
        st.rerun()

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=False):
            st.session_state.chat_history = []
            st.session_state.memory = ConversationMemory(max_turns=10)
            st.rerun()


# ════════════════════════════════════════════════════════════
#  PAGE: INSIGHTS
# ════════════════════════════════════════════════════════════

elif page == "💡 Insights":
    st.markdown("""
    <div class="fw-page-header">
        <h2>Financial Insights</h2>
        <p>AI-generated insights based on your spending patterns</p>
    </div>
    """, unsafe_allow_html=True)

    if not require_data():
        st.stop()

    insights = st.session_state.insights
    anomalies = st.session_state.anomalies
    recommendations = st.session_state.recommendations
    profile = st.session_state.user_profile

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### 🔍 Auto-Detected Insights")
        if insights:
            for insight in insights:
                sev = insight.get("severity", "info")
                icon = insight.get("icon", "📌")
                st.markdown(f"""
                <div class="fw-insight-card {sev}">
                    <div class="fw-insight-icon">{icon}</div>
                    <div>
                        <div class="fw-insight-title">{insight['title']}</div>
                        <div class="fw-insight-detail">{insight['detail']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant insights detected for your current data.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Anomaly section
        st.markdown("### 🚨 Anomaly Detection")
        if not anomalies.empty:
            st.markdown(f"""
            <div class="fw-insight-card warning">
                <div class="fw-insight-icon">⚠️</div>
                <div>
                    <div class="fw-insight-title">{len(anomalies)} Unusual Transactions Detected</div>
                    <div class="fw-insight-detail">Isolation Forest identified statistically abnormal transactions.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            for _, row in anomalies.iterrows():
                st.markdown(f"""
                <div class="fw-anomaly-row">
                    <span style="color:#8b9dc3;">{row['date'].strftime('%d %b %Y')}</span>
                    <span style="color:#e2e8f7;">{row['category']}</span>
                    <span class="fw-anomaly-amount">₹{row['amount']:,.0f}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="fw-insight-card success">
                <div class="fw-insight-icon">✅</div>
                <div>
                    <div class="fw-insight-title">No Anomalies Detected</div>
                    <div class="fw-insight-detail">All transactions appear consistent with your spending patterns.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### 💡 Personalized Recommendations")
        priority_colors = {"critical": "danger", "high": "warning", "medium": "info"}

        if recommendations:
            for rec in recommendations:
                color_cls = priority_colors.get(rec["priority"], "info")
                saving_str = f"+₹{rec['annual_saving']:,.0f}/yr" if rec["annual_saving"] > 0 else ""
                st.markdown(f"""
                <div class="fw-rec-card">
                    <div class="fw-rec-priority {rec['priority']}"></div>
                    <div style="flex:1;">
                        <div class="fw-rec-action">{rec['action']}</div>
                        <div style="font-size:0.78rem; color:#8b9dc3; margin-top:3px;">{rec['detail']}</div>
                    </div>
                    {f'<div class="fw-rec-saving">{saving_str}</div>' if saving_str else ''}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Spending heatmap by day of week
        st.markdown("### 📅 Spending by Day of Week")
        df = st.session_state.df
        expenses = df[df["type"] == "expense"]
        dow_spend = expenses.groupby("weekday")["amount"].sum()
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_spend = dow_spend.reindex([d for d in order if d in dow_spend.index])

        fig_dow = go.Figure(go.Bar(
            x=dow_spend.index.tolist(),
            y=dow_spend.values.tolist(),
            marker=dict(
                color=dow_spend.values.tolist(),
                colorscale=[[0, "#1f2d45"], [0.5, "#4f8ef7"], [1, "#ef4444"]],
                cornerradius=6
            ),
            hovertemplate="%{x}<br>₹%{y:,.0f}<extra></extra>",
            showlegend=False
        ))
        fig_dow.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=PLOTLY_AXIS,
            yaxis=PLOTLY_AXIS,
            height=240,
            showlegend=False
        )
        st.plotly_chart(fig_dow, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════
#  PAGE: PREDICTION
# ════════════════════════════════════════════════════════════

elif page == "📈 Prediction":
    st.markdown("""
    <div class="fw-page-header">
        <h2>Expense Forecasting</h2>
        <p>ML-powered predictions for your future spending</p>
    </div>
    """, unsafe_allow_html=True)

    if not require_data():
        st.stop()

    prediction = st.session_state.prediction
    cat_preds = st.session_state.cat_predictions
    summary = st.session_state.summary

    if "error" in prediction:
        st.warning(prediction["error"])
        st.stop()

    pred_val = prediction["next_month_prediction"]
    conf_low = prediction["confidence_range"]["low"]
    conf_high = prediction["confidence_range"]["high"]
    trend = prediction.get("trend", "stable")

    # Prediction KPIs
    st.markdown(f"""
    <div class="fw-prediction-summary">
        <div class="fw-prediction-stat">
            <div class="fw-prediction-stat-value">₹{pred_val:,.0f}</div>
            <div class="fw-prediction-stat-label">Predicted Next Month</div>
        </div>
        <div class="fw-prediction-stat">
            <div class="fw-prediction-stat-value" style="color:{'#ef4444' if trend=='increasing' else '#22c55e'};">
                {'📈' if trend=='increasing' else '📉'} {trend.capitalize()}
            </div>
            <div class="fw-prediction-stat-label">Spending Trend</div>
        </div>
        <div class="fw-prediction-stat">
            <div class="fw-prediction-stat-value" style="font-size:1.1rem;">₹{conf_low:,.0f}–{conf_high:,.0f}</div>
            <div class="fw-prediction-stat-label">90% Confidence Range</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Forecast Chart
    hist_months = prediction["historical_months"]
    hist_amounts = prediction["historical_amounts"]
    full_forecast = prediction["full_forecast"]

    n_hist = len(hist_months)
    n_future = len(full_forecast) - n_hist

    future_labels = []
    last_period = pd.Period(hist_months[-1], freq="M")
    for i in range(1, n_future + 1):
        future_labels.append(str(last_period + i))

    all_labels = hist_months + future_labels
    all_values = full_forecast

    fig_forecast = go.Figure()

    # Historical
    fig_forecast.add_trace(go.Scatter(
        x=hist_months,
        y=hist_amounts,
        name="Actual",
        mode="lines+markers",
        line=dict(color="#4f8ef7", width=2.5),
        marker=dict(size=8, color="#4f8ef7"),
        hovertemplate="%{x}<br>Actual: ₹%{y:,.0f}<extra></extra>"
    ))

    # Forecast
    fig_forecast.add_trace(go.Scatter(
        x=[hist_months[-1]] + future_labels,
        y=[hist_amounts[-1]] + all_values[n_hist:],
        name="Forecast",
        mode="lines+markers",
        line=dict(color="#a78bfa", width=2.5, dash="dash"),
        marker=dict(size=8, color="#a78bfa", symbol="diamond"),
        hovertemplate="%{x}<br>Forecast: ₹%{y:,.0f}<extra></extra>"
    ))

    # Confidence band
    fig_forecast.add_trace(go.Scatter(
        x=[future_labels[0]] + future_labels,
        y=[conf_high] * (len(future_labels) + 1),
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip"
    ))
    fig_forecast.add_trace(go.Scatter(
        x=[future_labels[0]] + future_labels,
        y=[conf_low] * (len(future_labels) + 1),
        mode="lines", line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(167,139,250,0.1)",
        name="Confidence Band",
        hoverinfo="skip"
    ))

    fig_forecast.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Spending Forecast", font=dict(size=14, color="#e2e8f7")),
        legend=PLOTLY_LEGEND,
        xaxis=PLOTLY_AXIS,
        yaxis=PLOTLY_AXIS,
        height=380,
        hovermode="x unified",
        xaxis_tickangle=-30
    )
    st.plotly_chart(fig_forecast, use_container_width=True, config={"displayModeBar": False})

    # Category-wise predictions
    st.markdown("### 🏷️ Category-wise Next Month Forecast")
    top_cats = dict(list(cat_preds.items())[:8])

    col1, col2 = st.columns([1.5, 1])
    with col1:
        fig_cat = go.Figure(go.Bar(
            x=list(top_cats.values()),
            y=list(top_cats.keys()),
            orientation="h",
            marker=dict(
                color=list(range(len(top_cats))),
                colorscale=[[0, "#4f8ef7"], [1, "#a78bfa"]],
                cornerradius=4
            ),
            text=[f"₹{v:,.0f}" for v in top_cats.values()],
            textposition="outside",
            textfont=dict(size=10, color="#8b9dc3"),
            hovertemplate="%{y}<br>₹%{x:,.0f}<extra></extra>"
        ))
        fig_cat.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=PLOTLY_AXIS,
            yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)", tickcolor="#8b9dc3"),
            height=320,
            showlegend=False
        )
        st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        total_pred = sum(top_cats.values())
        for cat, val in list(top_cats.items())[:6]:
            pct = val / total_pred * 100 if total_pred > 0 else 0
            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:4px;">
                    <span style="color:#e2e8f7;">{cat}</span>
                    <span style="color:#8b9dc3;">₹{val:,.0f}</span>
                </div>
                <div style="height:4px;background:#1f2d45;border-radius:9999px;overflow:hidden;">
                    <div style="width:{pct:.0f}%;height:100%;background:linear-gradient(90deg,#4f8ef7,#a78bfa);border-radius:9999px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: GOALS
# ════════════════════════════════════════════════════════════

elif page == "🎯 Goals":
    st.markdown("""
    <div class="fw-page-header">
        <h2>Goal Planning</h2>
        <p>Set savings targets and get a personalized action plan</p>
    </div>
    """, unsafe_allow_html=True)

    if not require_data():
        st.stop()

    summary = st.session_state.summary

    col_form, col_result = st.columns([1, 1.5])

    with col_form:
        st.markdown("### 🎯 Set Your Goal")
        goal_amount = st.number_input(
            "Savings Goal (₹)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=5000,
            format="%d"
        )
        goal_months = st.slider(
            "Target Timeline (months)",
            min_value=1, max_value=36, value=6
        )

        st.markdown(f"""
        <div style="padding:12px; background:rgba(79,142,247,0.08); border-radius:10px; border:1px solid rgba(79,142,247,0.2); margin:8px 0;">
            <div style="font-size:0.8rem; color:#8b9dc3;">Required monthly savings</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:700; color:#4f8ef7;">
                ₹{goal_amount / goal_months:,.0f}/mo
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📊 Generate Plan", use_container_width=True):
            result = plan_goal(summary, goal_amount, goal_months, st.session_state.df)
            st.session_state.goal_result = result

    with col_result:
        if st.session_state.goal_result:
            result = st.session_state.goal_result
            feasible = result["feasible"]
            gap = result["savings_gap"]
            current_save = result["current_monthly_savings"]
            req_save = result["required_monthly_savings"]
            progress = result["progress_pct"]

            # Feasibility badge
            feasibility_html = (
                '<span class="fw-goal-feasibility feasible">✅ Feasible</span>'
                if feasible else
                '<span class="fw-goal-feasibility difficult">⚠️ Challenging</span>'
            )

            st.markdown(f"""
            <div class="fw-goal-card">
                <div class="fw-goal-header">
                    <div class="fw-goal-title">₹{result['goal_amount']:,.0f} in {result['target_months']} months</div>
                    {feasibility_html}
                </div>
                <div style="font-size:0.8rem; color:#8b9dc3; margin-bottom:6px;">
                    Goal Progress (at current savings rate)
                </div>
                <div class="fw-progress-bar-track">
                    <div class="fw-progress-bar-fill" style="width:{progress:.0f}%;"></div>
                </div>
                <div style="font-size:0.78rem; color:#8b9dc3; text-align:right;">{progress:.0f}%</div>
                <div class="fw-goal-stats">
                    <div class="fw-goal-stat">
                        <div class="fw-goal-stat-value">₹{current_save:,.0f}</div>
                        <div class="fw-goal-stat-label">Current/mo</div>
                    </div>
                    <div class="fw-goal-stat">
                        <div class="fw-goal-stat-value" style="color:#4f8ef7;">₹{req_save:,.0f}</div>
                        <div class="fw-goal-stat-label">Required/mo</div>
                    </div>
                    <div class="fw-goal-stat">
                        <div class="fw-goal-stat-value" style="color:{'#ef4444' if gap > 0 else '#22c55e'};">₹{gap:,.0f}</div>
                        <div class="fw-goal-stat-label">Monthly Gap</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Suggested cuts
            if result["suggested_cuts"]:
                st.markdown("### ✂️ Suggested Expense Cuts")
                for cut in result["suggested_cuts"]:
                    st.markdown(f"""
                    <div class="fw-rec-card">
                        <div class="fw-rec-priority high"></div>
                        <div style="flex:1;">
                            <div class="fw-rec-action">Reduce {cut['category']}: ₹{cut['current']:,.0f} → ₹{cut['new_budget']:,.0f}/mo</div>
                            <div style="font-size:0.78rem; color:#8b9dc3;">Cut ₹{cut['suggested_cut']:,.0f}/month from this category</div>
                        </div>
                        <div class="fw-rec-saving">-₹{cut['suggested_cut']:,.0f}/mo</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Milestone chart
            if result["milestones"]:
                st.markdown("### 📈 Savings Milestone Projection")
                milestones = result["milestones"]
                fig_goal = go.Figure()
                fig_goal.add_trace(go.Scatter(
                    x=[m["month"] for m in milestones],
                    y=[m["accumulated"] for m in milestones],
                    mode="lines+markers",
                    name="Projected Savings",
                    line=dict(color="#22c55e", width=2.5),
                    marker=dict(size=7, color="#22c55e"),
                    fill="tozeroy",
                    fillcolor="rgba(34,197,94,0.08)",
                    hovertemplate="Month %{x}<br>₹%{y:,.0f} saved<extra></extra>"
                ))
                fig_goal.add_hline(
                    y=result["goal_amount"],
                    line_dash="dash",
                    line_color="#f59e0b",
                    annotation_text=f"Goal ₹{result['goal_amount']:,.0f}",
                    annotation_font_color="#f59e0b"
                )
                fig_goal.update_layout(
                    **PLOTLY_LAYOUT,
                    xaxis=PLOTLY_AXIS,
                    yaxis=PLOTLY_AXIS,
                    height=260,
                    xaxis_title="Month",
                    yaxis_title="Accumulated (₹)",
                    showlegend=False
                )
                st.plotly_chart(fig_goal, use_container_width=True, config={"displayModeBar": False})

        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem; background:rgba(79,142,247,0.04); border:1px dashed #1f2d45; border-radius:16px; margin-top:1rem;">
                <div style="font-size:2.5rem; margin-bottom:1rem;">🎯</div>
                <h3 style="color:#e2e8f7; font-family:'Syne',sans-serif; margin-bottom:0.5rem;">Set a Goal</h3>
                <p style="color:#8b9dc3; font-size:0.85rem;">Enter your savings target and timeline on the left to get a personalized plan.</p>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div class="fw-footer">
    FinWise AI — Intelligent Personal Finance Advisor &nbsp;
</div>
""", unsafe_allow_html=True)
