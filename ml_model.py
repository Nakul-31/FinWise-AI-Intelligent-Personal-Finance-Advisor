"""
FinWise AI — ML Prediction & Forecasting Engine
Linear Regression for expense forecasting, category-wise predictions,
and goal-based planning.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# ────────────────────────────────────────────────────────────
#  NEXT MONTH EXPENSE PREDICTION
# ────────────────────────────────────────────────────────────

def predict_next_month(df: pd.DataFrame) -> dict:
    """
    Predict total expenses for the next month using Linear Regression
    with polynomial features to capture trends.
    """
    expenses = df[df["type"] == "expense"]
    monthly = expenses.groupby("month")["amount"].sum().reset_index()
    monthly.columns = ["month", "total"]
    monthly = monthly.sort_values("month").reset_index(drop=True)

    if len(monthly) < 3:
        return {"error": "Need at least 3 months of data for prediction."}

    X = np.arange(len(monthly)).reshape(-1, 1)
    y = monthly["total"].values

    # Polynomial regression for trend capture
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    next_idx = np.array([[len(monthly)]])
    next_poly = poly.transform(next_idx)
    predicted = float(model.predict(next_poly)[0])
    predicted = max(predicted, 0)

    # Confidence estimate (std of residuals)
    residuals = y - model.predict(X_poly)
    std_err = float(np.std(residuals))

    mae = mean_absolute_error(y, model.predict(X_poly))

    # Generate forecast for plotting (historical + next 3 months)
    future_X = np.arange(len(monthly) + 3).reshape(-1, 1)
    future_poly = poly.transform(future_X)
    full_forecast = model.predict(future_poly).tolist()
    full_forecast = [max(v, 0) for v in full_forecast]

    return {
        "next_month_prediction": round(predicted, 2),
        "confidence_range": {
            "low": round(max(predicted - 1.5 * std_err, 0), 2),
            "high": round(predicted + 1.5 * std_err, 2)
        },
        "mae": round(mae, 2),
        "historical_months": monthly["month"].tolist(),
        "historical_amounts": monthly["total"].tolist(),
        "full_forecast": [round(v, 2) for v in full_forecast],
        "trend": "increasing" if full_forecast[-1] > full_forecast[-4] else "decreasing"
    }


# ────────────────────────────────────────────────────────────
#  CATEGORY-WISE PREDICTION
# ────────────────────────────────────────────────────────────

def predict_category_wise(df: pd.DataFrame) -> dict:
    """
    Predict next month's spending per category using Linear Regression.
    """
    expenses = df[df["type"] == "expense"]
    categories = expenses["category"].unique()
    predictions = {}

    for cat in categories:
        cat_monthly = expenses[expenses["category"] == cat].groupby("month")["amount"].sum().reset_index()
        cat_monthly = cat_monthly.sort_values("month").reset_index(drop=True)

        if len(cat_monthly) < 2:
            predictions[cat] = float(cat_monthly["amount"].mean()) if len(cat_monthly) > 0 else 0
            continue

        X = np.arange(len(cat_monthly)).reshape(-1, 1)
        y = cat_monthly["amount"].values
        model = LinearRegression()
        model.fit(X, y)
        pred = float(model.predict([[len(cat_monthly)]])[0])
        predictions[cat] = round(max(pred, 0), 2)

    return dict(sorted(predictions.items(), key=lambda x: -x[1]))


# ────────────────────────────────────────────────────────────
#  GOAL-BASED PLANNING
# ────────────────────────────────────────────────────────────

def plan_goal(
    summary: dict,
    goal_amount: float,
    months: int,
    df: pd.DataFrame = None
) -> dict:
    """
    Given a savings goal and timeline, compute:
    - Required monthly savings
    - Current deficit/surplus
    - Suggested expense cuts per category
    """
    monthly_income = summary["total_income"] / max(summary["months_covered"], 1)
    monthly_expense = summary["avg_monthly_expense"]
    current_monthly_savings = monthly_income - monthly_expense
    required_monthly_savings = goal_amount / months
    savings_gap = required_monthly_savings - current_monthly_savings

    feasible = savings_gap <= 0 or savings_gap < monthly_income * 0.5

    # Suggested cuts
    cuts = []
    if savings_gap > 0 and df is not None:
        expenses = df[df["type"] == "expense"]
        cat_monthly_avg = (
            expenses.groupby("category")["amount"].sum() / max(summary["months_covered"], 1)
        ).sort_values(ascending=False)

        discretionary = ["Food & Dining", "Shopping", "Entertainment", "Travel"]
        remaining_gap = savings_gap

        for cat in discretionary:
            if cat in cat_monthly_avg.index and remaining_gap > 0:
                current = cat_monthly_avg[cat]
                max_cut = current * 0.25  # suggest up to 25% cut
                actual_cut = min(max_cut, remaining_gap)
                if actual_cut > 100:
                    cuts.append({
                        "category": cat,
                        "current": round(current, 2),
                        "suggested_cut": round(actual_cut, 2),
                        "new_budget": round(current - actual_cut, 2)
                    })
                    remaining_gap -= actual_cut

    # Milestone timeline
    milestones = []
    for i in range(1, months + 1):
        accumulated = current_monthly_savings * i
        if savings_gap > 0:
            accumulated += sum(c["suggested_cut"] for c in cuts) * i
        accumulated = min(accumulated, goal_amount)
        milestones.append({"month": i, "accumulated": round(max(accumulated, 0), 2)})

    return {
        "goal_amount": goal_amount,
        "target_months": months,
        "required_monthly_savings": round(required_monthly_savings, 2),
        "current_monthly_savings": round(current_monthly_savings, 2),
        "current_monthly_income": round(monthly_income, 2),
        "current_monthly_expense": round(monthly_expense, 2),
        "savings_gap": round(max(savings_gap, 0), 2),
        "feasible": feasible,
        "suggested_cuts": cuts,
        "milestones": milestones,
        "progress_pct": round(
            min((current_monthly_savings * months / goal_amount) * 100, 100), 1
        ) if goal_amount > 0 else 0
    }


# ────────────────────────────────────────────────────────────
#  SPENDING TREND SUMMARY FOR LLM
# ────────────────────────────────────────────────────────────

def get_prediction_context(prediction: dict, cat_predictions: dict) -> str:
    """Format prediction data as LLM-readable context."""
    top5 = list(cat_predictions.items())[:5]
    cat_str = ", ".join(f"{k}: ₹{v:,.0f}" for k, v in top5)

    lines = [
        f"=== PREDICTION CONTEXT ===",
        f"Next Month Total Expense Forecast: ₹{prediction.get('next_month_prediction', 0):,.0f}",
        f"Confidence Range: ₹{prediction.get('confidence_range', {}).get('low', 0):,.0f} – ₹{prediction.get('confidence_range', {}).get('high', 0):,.0f}",
        f"Spending Trend: {prediction.get('trend', 'stable').capitalize()}",
        f"Forecast Accuracy (MAE): ₹{prediction.get('mae', 0):,.0f}",
        f"Category-wise Next Month Forecast: {cat_str}",
    ]
    return "\n".join(lines)
