"""
FinWise AI — Insights & Anomaly Detection Engine
Generates automatic financial insights and detects unusual transactions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


# ────────────────────────────────────────────────────────────
#  USER CLASSIFICATION
# ────────────────────────────────────────────────────────────

def classify_user(summary: dict) -> dict:
    """
    Classify user as Saver / Balanced / Spender / Risky
    based on savings rate and spending patterns.
    """
    total_income = summary["total_income"]
    total_expense = summary["total_expense"]
    savings_rate = ((total_income - total_expense) / total_income * 100) if total_income > 0 else 0

    mom_change = summary["month_over_month_change_pct"]

    if savings_rate >= 30:
        profile = "Saver"
        color = "green"
        emoji = "💚"
        description = "Excellent! You save more than 30% of income."
    elif savings_rate >= 15:
        profile = "Balanced"
        color = "blue"
        emoji = "💙"
        description = "You maintain a healthy savings rate."
    elif savings_rate >= 0:
        profile = "Spender"
        color = "orange"
        emoji = "🟠"
        description = "You're spending most of your income. Consider reducing discretionary expenses."
    else:
        profile = "Risky"
        color = "red"
        emoji = "🔴"
        description = "You're spending more than you earn. Immediate action recommended."

    # Check spending trend
    trend_alert = None
    if mom_change > 15:
        trend_alert = f"⚠️ Spending increased by {mom_change:.1f}% last month."
    elif mom_change < -10:
        trend_alert = f"✅ Spending decreased by {abs(mom_change):.1f}% last month."

    return {
        "profile": profile,
        "color": color,
        "emoji": emoji,
        "description": description,
        "savings_rate": round(savings_rate, 1),
        "trend_alert": trend_alert,
        "mom_change": mom_change
    }


# ────────────────────────────────────────────────────────────
#  INSIGHT GENERATION
# ────────────────────────────────────────────────────────────

def generate_insights(df: pd.DataFrame, summary: dict) -> list[dict]:
    """
    Generate a list of actionable, data-driven financial insights.
    Each insight has: type, title, detail, severity (info/warning/danger/success)
    """
    insights = []
    expenses = df[df["type"] == "expense"]

    # 1. Month-over-month spending change
    mom = summary["month_over_month_change_pct"]
    if abs(mom) >= 5:
        sev = "danger" if mom > 15 else ("warning" if mom > 5 else "success")
        insights.append({
            "type": "trend",
            "title": f"Spending {'Up' if mom > 0 else 'Down'} {abs(mom):.1f}% MoM",
            "detail": (
                f"You spent ₹{summary['last_month_expense']:,.0f} last month vs "
                f"₹{summary['prev_month_expense']:,.0f} the month before."
            ),
            "severity": sev,
            "icon": "📈" if mom > 0 else "📉"
        })

    # 2. Top spending category alert
    cat_breakdown = summary["category_breakdown_last_month"]
    if cat_breakdown:
        top_cat = max(cat_breakdown, key=cat_breakdown.get)
        top_val = cat_breakdown[top_cat]
        total_last = summary["last_month_expense"]
        pct = (top_val / total_last * 100) if total_last > 0 else 0
        sev = "warning" if pct > 40 else "info"
        insights.append({
            "type": "category",
            "title": f"{top_cat} is Your Biggest Expense",
            "detail": f"₹{top_val:,.0f} ({pct:.1f}% of last month's total spending).",
            "severity": sev,
            "icon": "🏷️"
        })

    # 3. Weekend vs Weekday
    wkend = summary["weekend_spend"]
    wkday = summary["weekday_spend"]
    if wkday > 0:
        wkend_pct = wkend / (wkend + wkday) * 100
        if wkend_pct > 35:
            insights.append({
                "type": "behavior",
                "title": f"High Weekend Spending ({wkend_pct:.0f}% of total)",
                "detail": f"You spend ₹{wkend:,.0f} on weekends vs ₹{wkday:,.0f} on weekdays. Consider a weekend budget.",
                "severity": "warning",
                "icon": "📅"
            })

    # 4. Savings rate insight
    savings_rate = (summary["net_savings"] / summary["total_income"] * 100) if summary["total_income"] > 0 else 0
    if savings_rate < 10:
        insights.append({
            "type": "savings",
            "title": f"Low Savings Rate: {savings_rate:.1f}%",
            "detail": "Financial experts recommend saving at least 20% of income. Try the 50/30/20 rule.",
            "severity": "danger",
            "icon": "💸"
        })
    elif savings_rate >= 30:
        insights.append({
            "type": "savings",
            "title": f"Excellent Savings Rate: {savings_rate:.1f}%",
            "detail": f"You're saving ₹{summary['net_savings']:,.0f} total. Consider investing the surplus.",
            "severity": "success",
            "icon": "🏆"
        })

    # 5. Category comparison (highest growth)
    if len(summary["months_list"]) >= 2:
        last_m = summary["months_list"][-1]
        prev_m = summary["months_list"][-2]
        last_cat = expenses[expenses["month"] == last_m].groupby("category")["amount"].sum()
        prev_cat = expenses[expenses["month"] == prev_m].groupby("category")["amount"].sum()
        combined = pd.DataFrame({"last": last_cat, "prev": prev_cat}).dropna()
        combined["change_pct"] = ((combined["last"] - combined["prev"]) / combined["prev"] * 100)
        if not combined.empty:
            max_growth = combined["change_pct"].idxmax()
            max_pct = combined.loc[max_growth, "change_pct"]
            if max_pct > 20:
                insights.append({
                    "type": "category_growth",
                    "title": f"{max_growth} Spending Surged {max_pct:.0f}%",
                    "detail": f"₹{combined.loc[max_growth,'prev']:,.0f} → ₹{combined.loc[max_growth,'last']:,.0f} month-over-month.",
                    "severity": "warning",
                    "icon": "🚨"
                })

    # 6. Large single transaction
    max_txn = expenses["amount"].max()
    if max_txn > summary["avg_monthly_expense"] * 0.3:
        max_row = expenses.loc[expenses["amount"].idxmax()]
        insights.append({
            "type": "large_transaction",
            "title": f"Large Transaction: ₹{max_txn:,.0f}",
            "detail": f"Category: {max_row['category']} on {max_row['date'].strftime('%d %b %Y')}.",
            "severity": "info",
            "icon": "💰"
        })

    return insights


# ────────────────────────────────────────────────────────────
#  ANOMALY DETECTION
# ────────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Use Isolation Forest to flag statistically unusual transactions.
    Returns a DataFrame of anomalous expense records.
    """
    expenses = df[df["type"] == "expense"].copy()
    if len(expenses) < 10:
        return pd.DataFrame()

    # Features: amount + day of month + day of week
    features = pd.DataFrame({
        "amount": expenses["amount"],
        "day_of_month": expenses["date"].dt.day,
        "day_of_week": expenses["date"].dt.dayofweek,
    })

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    preds = model.fit_predict(features)
    expenses["is_anomaly"] = preds == -1
    expenses["anomaly_score"] = model.score_samples(features)

    anomalies = expenses[expenses["is_anomaly"]].copy()
    anomalies = anomalies.sort_values("anomaly_score").head(10)
    return anomalies[["date", "description", "category", "amount", "anomaly_score"]].reset_index(drop=True)


# ────────────────────────────────────────────────────────────
#  RECOMMENDATION ENGINE
# ────────────────────────────────────────────────────────────

def generate_recommendations(df: pd.DataFrame, summary: dict, user_profile: dict) -> list[dict]:
    """Generate personalized, data-driven spending recommendations."""
    recommendations = []
    expenses = df[df["type"] == "expense"]

    # Category-specific recommendations
    discretionary = ["Food & Dining", "Entertainment", "Shopping", "Travel"]
    cat_all = summary["category_breakdown_all_time"]
    months = max(summary["months_covered"], 1)

    for cat in discretionary:
        if cat in cat_all:
            monthly_avg = cat_all[cat] / months
            if cat == "Food & Dining" and monthly_avg > 10000:
                cut = round(monthly_avg * 0.2, -2)
                recommendations.append({
                    "category": cat,
                    "action": f"Reduce {cat} by ₹{cut:,.0f}/month",
                    "detail": f"Current avg: ₹{monthly_avg:,.0f}/month. Cook at home 3–4 days/week.",
                    "annual_saving": round(cut * 12, -2),
                    "priority": "high"
                })
            elif cat == "Entertainment" and monthly_avg > 4000:
                cut = round(monthly_avg * 0.3, -2)
                recommendations.append({
                    "category": cat,
                    "action": f"Trim {cat} to ₹{monthly_avg - cut:,.0f}/month",
                    "detail": "Cancel unused streaming subscriptions. Look for free events.",
                    "annual_saving": round(cut * 12, -2),
                    "priority": "medium"
                })
            elif cat == "Shopping" and monthly_avg > 7000:
                cut = round(monthly_avg * 0.25, -2)
                recommendations.append({
                    "category": cat,
                    "action": f"Apply a 48-hour rule before purchases over ₹{cut:,.0f}",
                    "detail": f"Current avg: ₹{monthly_avg:,.0f}/month. Impulse purchases are costing you.",
                    "annual_saving": round(cut * 12, -2),
                    "priority": "medium"
                })

    # Investment recommendation
    savings_rate = user_profile["savings_rate"]
    if savings_rate > 15 and "Investments" not in cat_all:
        recommendations.append({
            "category": "Investments",
            "action": "Start a SIP with ₹5,000/month",
            "detail": "You have surplus savings. A ₹5k/month SIP in index funds can build ₹10L+ in 10 years.",
            "annual_saving": 60000,
            "priority": "high"
        })

    # Emergency fund
    avg_monthly = summary["avg_monthly_expense"]
    if summary["net_savings"] < avg_monthly * 3:
        recommendations.append({
            "category": "Emergency Fund",
            "action": f"Build an emergency fund of ₹{avg_monthly * 3:,.0f}",
            "detail": "Target 3 months of expenses as a safety net before investing.",
            "annual_saving": 0,
            "priority": "critical"
        })

    return sorted(recommendations, key=lambda x: {"critical": 0, "high": 1, "medium": 2}.get(x["priority"], 3))
