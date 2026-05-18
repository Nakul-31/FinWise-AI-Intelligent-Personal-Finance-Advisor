# 💼 FinWise AI — Intelligent Personal Finance Advisor

A production-ready AI-powered fintech chatbot that acts as your personal financial advisor — grounded entirely in your real financial data.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the application
```bash
streamlit run app.py
```

### 3. Open in browser
```
http://localhost:8501
```

---

## 🏗️ Project Structure

```
finwise_ai/
├── app.py                  # Main Streamlit application (all pages + routing)
├── styles.css              # Complete design system (CSS variables + components)
├── data_processing.py      # Data ingestion, parsing, financial summary builder
├── insights.py             # Insight generation, anomaly detection, recommendations
├── ml_model.py             # Prediction engine (Linear Regression + forecasting)
├── chatbot.py              # Conversation memory, intent classification, prompt builder
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🧠 Core Features

| Feature | Description |
|---|---|
| **AI Chatbot** | Claude-powered advisor grounded in your real financial data |
| **Dashboard** | KPI cards, trend charts, category breakdown, income vs expense |
| **Insights** | Auto-generated spending insights, anomaly detection |
| **Prediction** | ML forecast for next month's total + category-wise spending |
| **Goal Planning** | Set savings targets, get personalized cut recommendations |
| **Conversational Memory** | Context-aware multi-turn conversations |

---

## 📋 CSV Format

Your CSV can have these columns (column names are auto-detected):

| Column | Required | Description |
|---|---|---|
| `date` | ✅ | Transaction date (any format) |
| `amount` | ✅ | Transaction amount in ₹ |
| `category` | Optional | Expense category |
| `type` | Optional | `income` or `expense` |
| `description` | Optional | Transaction notes |

---

## 🎨 Design System

The app uses a dark premium fintech aesthetic with:

- **Fonts**: Syne (display) + DM Sans (body)
- **Colors**: Deep navy background with electric blue accents
- **Semantic colors**: Green (savings) · Red (overspending) · Blue (insights)
- **CSS Variables**: Full design token system in `styles.css`

---

## 🤖 Chatbot Capabilities

Ask FinWise AI questions like:
- *"How much did I spend last month?"*
- *"Where am I overspending?"*
- *"Compare this month vs last month"*
- *"Save ₹1,00,000 in 6 months — is it possible?"*
- *"What's my biggest expense category?"*
- *"Detect any unusual transactions"*
- *"Am I a saver or spender?"*
- *"Give me personalized recommendations"*

All responses are grounded in your real transaction data — no generic AI answers.

---

## 🔑 API Key

The app uses the Anthropic Claude API. The API key is handled automatically when running inside Claude.ai artifacts. For standalone deployment, add your key:

```python
# In app.py call_claude(), add to headers:
"x-api-key": "YOUR_ANTHROPIC_API_KEY"
```

---

## 📦 Sample Data

Click **"Load Sample Data"** in the sidebar to generate 6 months of realistic synthetic financial data and explore all features immediately.
