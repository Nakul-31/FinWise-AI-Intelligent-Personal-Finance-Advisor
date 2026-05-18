"""
FinWise AI — Conversational Financial Advisor
Uses Anthropic Claude API directly with conversation memory.
All responses are grounded in user's financial data.
"""

import json
import re
from datetime import datetime


# ────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are FinWise AI — an expert personal finance advisor with access to the user's REAL financial data.

CRITICAL RULES:
1. NEVER give generic financial advice. ALL responses must reference specific numbers from the user's data.
2. Always quote exact rupee amounts from the data context provided.
3. Be concise, insightful, and direct. Max 150 words per response unless doing detailed analysis.
4. Use ₹ symbol for all amounts. Format large numbers with commas (e.g., ₹1,20,000).
5. Provide actionable recommendations — not vague tips.
6. If asked about something not in the data, say so clearly.
7. Use emojis sparingly to highlight key insights (📈 📉 ✅ ⚠️ 💡).
8. For comparisons, always give percentage AND absolute values.

RESPONSE FORMAT:
- Start with the direct answer (1-2 sentences)
- Follow with 2-3 key data points
- End with 1 actionable recommendation

You have access to:
- Transaction history with categories, amounts, dates
- Monthly spending trends
- Income vs expense analysis
- ML-generated predictions
- User behavior profile
"""

# ────────────────────────────────────────────────────────────
#  SUGGESTED PROMPTS
# ────────────────────────────────────────────────────────────

SUGGESTED_PROMPTS = [
    "How much did I spend last month?",
    "Where am I overspending?",
    "Compare this month vs last month",
    "How can I save ₹50,000 in 6 months?",
    "What's my biggest expense category?",
    "Predict my next month expenses",
    "Am I a saver or spender?",
    "Detect any unusual transactions",
    "What's my savings rate?",
    "Give me personalized recommendations",
]


# ────────────────────────────────────────────────────────────
#  CONVERSATION MANAGER
# ────────────────────────────────────────────────────────────

class ConversationMemory:
    """Maintains sliding window conversation history."""

    def __init__(self, max_turns: int = 10):
        self.history = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        # Sliding window — keep last N turns
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def get_messages(self) -> list:
        return self.history.copy()

    def clear(self):
        self.history = []

    def to_json(self) -> str:
        return json.dumps(self.history)

    @classmethod
    def from_json(cls, data: str, max_turns: int = 10) -> "ConversationMemory":
        mem = cls(max_turns=max_turns)
        mem.history = json.loads(data)
        return mem


# ────────────────────────────────────────────────────────────
#  INTENT CLASSIFIER (rule-based pre-filter)
# ────────────────────────────────────────────────────────────

def classify_intent(query: str) -> str:
    """Quick rule-based intent detection to route context."""
    q = query.lower()

    if any(w in q for w in ["predict", "next month", "forecast", "future"]):
        return "prediction"
    elif any(w in q for w in ["anomal", "unusual", "suspicious", "weird", "outlier"]):
        return "anomaly"
    elif any(w in q for w in ["save", "goal", "target", "plan", "months", "lakh", "lakhs"]):
        return "goal"
    elif any(w in q for w in ["insight", "recommend", "suggest", "advice", "tip", "improve"]):
        return "recommendation"
    elif any(w in q for w in ["compare", "vs", "versus", "difference", "change"]):
        return "comparison"
    elif any(w in q for w in ["spend", "spent", "expense", "cost", "pay", "paid"]):
        return "spending"
    elif any(w in q for w in ["income", "earn", "salary", "revenue"]):
        return "income"
    elif any(w in q for w in ["saver", "spender", "profile", "type", "classify"]):
        return "profile"
    else:
        return "general"


# ────────────────────────────────────────────────────────────
#  CONTEXT BUILDER
# ────────────────────────────────────────────────────────────

def build_user_message(
    query: str,
    financial_context: str,
    prediction_context: str,
    insights: list,
    user_profile: dict,
    anomalies_summary: str = "",
    recommendations: list = None
) -> str:
    """
    Build a rich user message with all relevant financial context
    injected inline so the LLM can give data-grounded responses.
    """
    intent = classify_intent(query)

    # Always include core financial context
    context_parts = [financial_context]

    # Intent-based context injection
    if intent in ("prediction", "general"):
        context_parts.append(prediction_context)

    if intent in ("anomaly", "general"):
        if anomalies_summary:
            context_parts.append(f"=== ANOMALY CONTEXT ===\n{anomalies_summary}")

    if intent in ("recommendation", "profile", "general"):
        if recommendations:
            rec_str = "\n".join(
                f"- {r['action']} (saves ₹{r['annual_saving']:,.0f}/yr)" if r["annual_saving"] > 0
                else f"- {r['action']}"
                for r in (recommendations or [])[:5]
            )
            context_parts.append(f"=== RECOMMENDATIONS ===\n{rec_str}")

        profile_str = (
            f"User Profile: {user_profile.get('profile', 'N/A')} | "
            f"Savings Rate: {user_profile.get('savings_rate', 0):.1f}%"
        )
        context_parts.append(profile_str)

    if intent in ("general", "spending") and insights:
        insight_str = "\n".join(
            f"- [{i.get('severity','info').upper()}] {i.get('title','')}: {i.get('detail','')}"
            for i in insights[:4]
        )
        context_parts.append(f"=== KEY INSIGHTS ===\n{insight_str}")

    full_context = "\n\n".join(context_parts)

    return f"""{full_context}

---
USER QUESTION: {query}

Answer using ONLY the data above. Be specific with numbers."""


# ────────────────────────────────────────────────────────────
#  GOAL PARSER
# ────────────────────────────────────────────────────────────

def parse_goal_from_query(query: str) -> tuple[float | None, int | None]:
    """
    Extract goal amount and months from natural language.
    e.g. "Save 1 lakh in 6 months" → (100000, 6)
    """
    query_lower = query.lower()

    # Amount parsing
    amount = None
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*lakh", query_lower)
    if lakh_match:
        amount = float(lakh_match.group(1)) * 100000

    if amount is None:
        k_match = re.search(r"₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*k\b", query_lower)
        if k_match:
            amount = float(k_match.group(1).replace(",", "")) * 1000

    if amount is None:
        num_match = re.search(r"₹?\s*(\d+(?:,\d+)+(?:\.\d+)?)", query_lower)
        if num_match:
            amount = float(num_match.group(1).replace(",", ""))

    if amount is None:
        plain_match = re.search(r"save\s+(\d+(?:,\d+)*)", query_lower)
        if plain_match:
            amount = float(plain_match.group(1).replace(",", ""))

    # Month parsing
    months = None
    month_match = re.search(r"(\d+)\s*month", query_lower)
    if month_match:
        months = int(month_match.group(1))

    year_match = re.search(r"(\d+)\s*year", query_lower)
    if year_match and months is None:
        months = int(year_match.group(1)) * 12

    return amount, months
