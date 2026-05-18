"""
FinWise AI — Data Processing Engine
Handles all financial data ingestion, cleaning, and structuring.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json


# ────────────────────────────────────────────────────────────
#  SAMPLE DATA GENERATOR
# ────────────────────────────────────────────────────────────

CATEGORIES = [
    "Food & Dining", "Shopping", "Transport", "Entertainment",
    "Healthcare", "Utilities", "Rent", "Education", "Travel", "Investments"
]

INCOME_SOURCES = ["Salary", "Freelance", "Dividends", "Rental Income"]


def generate_sample_data(months: int = 6, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic transaction data."""
    random.seed(seed)
    np.random.seed(seed)

    records = []
    base_date = datetime.now() - timedelta(days=months * 30)

    # Monthly income
    monthly_salary = random.randint(60000, 90000)

    for month_offset in range(months):
        month_start = base_date + timedelta(days=month_offset * 30)

        # Income entries
        records.append({
            "date": month_start + timedelta(days=1),
            "description": "Monthly Salary",
            "category": "Salary",
            "amount": monthly_salary + random.randint(-3000, 3000),
            "type": "income"
        })

        if random.random() > 0.5:
            records.append({
                "date": month_start + timedelta(days=random.randint(5, 25)),
                "description": "Freelance Project",
                "category": "Freelance",
                "amount": random.randint(5000, 20000),
                "type": "income"
            })

        # Expense entries — varied by category
        category_budgets = {
            "Food & Dining": (8000, 14000, 25),
            "Shopping": (3000, 9000, 12),
            "Transport": (1500, 4000, 18),
            "Entertainment": (1000, 4000, 6),
            "Healthcare": (0, 3000, 3),
            "Utilities": (1500, 2500, 4),
            "Rent": (15000, 20000, 1),
            "Education": (0, 5000, 2),
            "Travel": (0, 12000, 2),
            "Investments": (2000, 8000, 2)
        }

        for cat, (lo, hi, txn_count) in category_budgets.items():
            total = random.randint(lo, hi)
            if total == 0:
                continue
            for _ in range(txn_count):
                day = random.randint(0, 29)
                records.append({
                    "date": month_start + timedelta(days=day),
                    "description": f"{cat} expense",
                    "category": cat,
                    "amount": round(total / txn_count + random.uniform(-200, 200), 2),
                    "type": "expense"
                })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["amount"] = df["amount"].abs()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["weekday"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    return df


# ────────────────────────────────────────────────────────────
#  UPLOADED CSV PARSER
# ────────────────────────────────────────────────────────────

def parse_uploaded_csv(filepath: str) -> pd.DataFrame:
    """Parse user-uploaded CSV with flexible column mapping."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Remove duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    # If the CSV already has all required columns, use them directly
    required = {"date", "amount", "category", "type", "description"}
    if required.issubset(set(df.columns)):
        # Already in the right format — just parse types
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").abs()
        df.dropna(subset=["date", "amount"], inplace=True)
        # Regenerate derived columns (in case they're missing or stale)
        df["month"] = df["date"].dt.to_period("M").astype(str)
        df["weekday"] = df["date"].dt.day_name()
        df["is_weekend"] = df["date"].dt.dayofweek >= 5
        return df.reset_index(drop=True)

    # Flexible column mapping for unknown CSV formats
    # Map each column once only — priority order ensures no duplicates
    mapped = {}
    for col in df.columns:
        if "date" in col and "date" not in mapped.values():
            mapped[col] = "date"
        elif ("amount" in col or "debit" in col or "credit" in col) and "amount" not in mapped.values():
            mapped[col] = "amount"
        elif "category" in col and "category" not in mapped.values():
            mapped[col] = "category"
        elif col == "type" and "type" not in mapped.values():
            mapped[col] = "type"
        elif ("desc" in col or "narr" in col or "detail" in col) and "description" not in mapped.values():
            mapped[col] = "description"

    df.rename(columns=mapped, inplace=True)

    # Validate required columns
    for req in ["date", "amount"]:
        if req not in df.columns:
            raise ValueError(f"Missing required column: '{req}'")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").abs()
    df.dropna(subset=["date", "amount"], inplace=True)

    if "category" not in df.columns:
        df["category"] = "Uncategorized"
    if "type" not in df.columns:
        df["type"] = "expense"
    if "description" not in df.columns:
        df["description"] = df["category"]

    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["weekday"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    return df.reset_index(drop=True)


# ────────────────────────────────────────────────────────────
#  FINANCIAL SUMMARY BUILDER
# ────────────────────────────────────────────────────────────

def build_financial_summary(df: pd.DataFrame) -> dict:
    """Build a structured financial summary for the LLM context."""
    expenses = df[df["type"] == "expense"]
    income = df[df["type"] == "income"]

    # Monthly aggregates
    monthly_expenses = expenses.groupby("month")["amount"].sum().round(2)
    monthly_income = income.groupby("month")["amount"].sum().round(2)

    months_list = sorted(monthly_expenses.index.tolist())
    last_month = months_list[-1] if months_list else None
    prev_month = months_list[-2] if len(months_list) > 1 else None

    last_month_expense = float(monthly_expenses.get(last_month, 0))
    prev_month_expense = float(monthly_expenses.get(prev_month, 0)) if prev_month else 0

    pct_change = 0.0
    if prev_month_expense > 0:
        pct_change = round(((last_month_expense - prev_month_expense) / prev_month_expense) * 100, 1)

    # Category breakdown (last month)
    if last_month:
        cat_last = expenses[expenses["month"] == last_month].groupby("category")["amount"].sum().round(2)
    else:
        cat_last = expenses.groupby("category")["amount"].sum().round(2)

    top_category = cat_last.idxmax() if not cat_last.empty else "N/A"

    # Weekend vs Weekday
    weekend_spend = float(expenses[expenses["is_weekend"]]["amount"].sum())
    weekday_spend = float(expenses[~expenses["is_weekend"]]["amount"].sum())

    # Total stats
    total_expense = float(expenses["amount"].sum())
    total_income = float(income["amount"].sum())
    avg_monthly_expense = float(monthly_expenses.mean()) if not monthly_expenses.empty else 0

    summary = {
        "total_expense": round(total_expense, 2),
        "total_income": round(total_income, 2),
        "net_savings": round(total_income - total_expense, 2),
        "avg_monthly_expense": round(avg_monthly_expense, 2),
        "last_month": last_month,
        "last_month_expense": last_month_expense,
        "prev_month": prev_month,
        "prev_month_expense": prev_month_expense,
        "month_over_month_change_pct": pct_change,
        "top_spending_category": top_category,
        "category_breakdown_last_month": cat_last.to_dict(),
        "category_breakdown_all_time": expenses.groupby("category")["amount"].sum().round(2).to_dict(),
        "monthly_expense_trend": monthly_expenses.to_dict(),
        "monthly_income_trend": monthly_income.to_dict(),
        "weekend_spend": round(weekend_spend, 2),
        "weekday_spend": round(weekday_spend, 2),
        "total_transactions": len(df),
        "months_covered": len(months_list),
        "months_list": months_list
    }
    return summary


def get_context_string(summary: dict) -> str:
    """Convert summary dict into a compact LLM-readable string."""
    cat_str = ", ".join(
        f"{k}: ₹{v:,.0f}"
        for k, v in sorted(summary["category_breakdown_last_month"].items(), key=lambda x: -x[1])
    )
    trend_str = ", ".join(
        f"{m}: ₹{v:,.0f}" for m, v in list(summary["monthly_expense_trend"].items())[-4:]
    )
    lines = [
        f"=== FINANCIAL DATA CONTEXT ===",
        f"Period Covered: {summary['months_covered']} months",
        f"Total Income: ₹{summary['total_income']:,.0f}",
        f"Total Expenses: ₹{summary['total_expense']:,.0f}",
        f"Net Savings: ₹{summary['net_savings']:,.0f}",
        f"Avg Monthly Expense: ₹{summary['avg_monthly_expense']:,.0f}",
        f"Last Month ({summary['last_month']}) Expense: ₹{summary['last_month_expense']:,.0f}",
        f"Previous Month ({summary['prev_month']}) Expense: ₹{summary['prev_month_expense']:,.0f}",
        f"Month-over-Month Change: {summary['month_over_month_change_pct']:+.1f}%",
        f"Top Spending Category: {summary['top_spending_category']}",
        f"Category Breakdown (Last Month): {cat_str}",
        f"Monthly Trend: {trend_str}",
        f"Weekend Spend: ₹{summary['weekend_spend']:,.0f} | Weekday Spend: ₹{summary['weekday_spend']:,.0f}",
        f"Total Transactions: {summary['total_transactions']}",
    ]
    return "\n".join(lines)