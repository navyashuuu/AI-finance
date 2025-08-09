
# ai.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sentence_transformers import SentenceTransformer, util
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import random
from PIL import Image
import base64
from fpdf import FPDF
import io
from io import BytesIO

# Set Streamlit config
st.set_page_config(page_title="AI Finance Manager", layout="wide")

@st.cache_data
def get_base64_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((500, 300))  # Resize to reduce weight
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    except Exception as e:
        st.warning(f"🚫 Image load error: {e}")
        return ""

def show_welcome():
    img_str = get_base64_image(r"C:\Users\Navya\Downloads\futuristic-robot-interacting-with-money.jpg")

    if not img_str:
        return

    st.markdown(
        f"""
        <style>
        .welcome-container {{
            animation: fadeIn 1.5s ease-in-out;
            text-align: center;
            padding: 20px;
            font-family: 'Segoe UI', sans-serif;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .glass-image {{
            display: inline-block;
            width: 500px;
            padding: 16px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.05);
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }}
        </style>

        <div class="welcome-container">
            <div class="glass-image">
                <img src="data:image/png;base64,{img_str}" width="100%" style="border-radius: 10px;" />
            </div>
            <h1 style="color: #00FFAB; font-size: 40px; font-weight: bold; margin-top: 25px;">
                💵 Welcome to <span style="color:#FFD700;">AI Finance Manager</span> 💵
            </h1>
            <p style="font-size: 18px; color: #ccc;">Your Intelligent Partner in Financial Planning</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Show only once during session (prevents reloading repeatedly)
if "welcome_shown" not in st.session_state:
    show_welcome()
    st.session_state.welcome_shown = True

# ───────────── Multilingual Support Setup ─────────────
langs = {"English": "en", "हिंदी (Hindi)": "hi", "తెలుగు (Telugu)": "te"}
language = st.selectbox("🌐 Select Language", list(langs.keys()))

lang_mappings = {
    "hi": {
        "खाना": "groceries", "भाड़ा": "rent", "यात्रा": "bus", "शॉपिंग": "shopping", "मोबाइल": "mobile bill",
        "इंटरनेट": "internet", "चिकित्सा": "hospital", "बिजली": "electricity bill"
    },
    "te": {
        "ఆహారం": "groceries", "అద్దె": "rent", "ప్రయాణం": "bus", "షాపింగ్": "shopping", "మొబైల్": "mobile bill",
        "ఇంటర్నెట్": "internet", "వైద్యం": "hospital", "కరెంట్": "electricity bill"
    }
}

def translate_expenses(text):
    lang_code = langs[language]
    if lang_code not in lang_mappings:
        return text
    for word, eng_word in lang_mappings[lang_code].items():
        text = text.replace(word, eng_word)
    return text

# ───────────── Daily AI Finance Tip Feed ─────────────
tips_calendar = {
    0: ("🏠 Rent Tip", "Pay rent first to secure your shelter and avoid late fees."),
    1: ("🍱 Food Tip", "Plan meals weekly to avoid impulsive food delivery spending."),
    2: ("🚌 Travel Tip", "Combine errands to save fuel or consider carpooling."),
    3: ("🛍️ Shopping Tip", "Wait 24 hours before buying non-essential items."),
    4: ("📱 Mobile Tip", "Monitor data usage to avoid unnecessary recharges."),
    5: ("🏥 Health Tip", "Maintain a small emergency health fund."),
    6: ("💡 Electricity Tip", "Unplug unused devices to reduce electricity costs."),
}
today = datetime.today().date()
weekday = today.weekday()
default_title, default_tip = tips_calendar.get(weekday, ("💡 Finance Tip", "Spend wisely."))

localized_tips = {
    "hi": {
        "🏠 Rent Tip": "🏠 किराया सलाह",
        "Pay rent first to secure your shelter and avoid late fees.": "सबसे पहले किराया चुकाएं ताकि रहने की जगह सुनिश्चित हो और देर से शुल्क से बचें।",
        "🍱 Food Tip": "🍱 भोजन सलाह",
        "Plan meals weekly to avoid impulsive food delivery spending.": "बिना योजना के खाना मंगवाने से बचने के लिए साप्ताहिक भोजन योजना बनाएं।",
        "🚌 Travel Tip": "🚌 यात्रा सलाह",
        "Combine errands to save fuel or consider carpooling.": "ईंधन बचाने के लिए काम एकसाथ करें या कार पूल करें।",
        "🛍️ Shopping Tip": "🛍️ खरीदारी सलाह",
        "Wait 24 hours before buying non-essential items.": "गैर-ज़रूरी वस्तुओं को खरीदने से पहले 24 घंटे रुकें।",
        "📱 Mobile Tip": "📱 मोबाइल सलाह",
        "Monitor data usage to avoid unnecessary recharges.": "अनावश्यक रिचार्ज से बचने के लिए डेटा उपयोग पर नज़र रखें।",
        "🏥 Health Tip": "🏥 स्वास्थ्य सलाह",
        "Maintain a small emergency health fund.": "एक छोटा स्वास्थ्य आपातकालीन फंड रखें।",
        "💡 Electricity Tip": "💡 बिजली सलाह",
        "Unplug unused devices to reduce electricity costs.": "बिजली बचाने के लिए बिना उपयोग वाले उपकरणों को अनप्लग करें।"
    },
    "te": {
        "🏠 Rent Tip": "🏠 అద్దె సూచన",
        "Pay rent first to secure your shelter and avoid late fees.": "తిరస్కరణకు ముందుగా అద్దె చెల్లించండి, ఆలస్య ఛార్జీలను నివారించండి.",
        "🍱 Food Tip": "🍱 ఆహార సూచన",
        "Plan meals weekly to avoid impulsive food delivery spending.": "ఆహారం ఆర్డర్ చేసే ఖర్చులను నివారించేందుకు వారానికి మెనూ సిద్ధం చేయండి.",
        "🚌 Travel Tip": "🚌 ప్రయాణ సూచన",
        "Combine errands to save fuel or consider carpooling.": "ఇంధనం ఆదా చేయడానికి పనులు కలిపి చేయండి లేదా కార్‌పూల్ పరిగణించండి.",
        "🛍️ Shopping Tip": "🛍️ షాపింగ్ సూచన",
        "Wait 24 hours before buying non-essential items.": "అవసరమేమీ లేని వస్తువులు కొనడంలో 24 గంటలు ఆలోచించండి.",
        "📱 Mobile Tip": "📱 మొబైల్ సూచన",
        "Monitor data usage to avoid unnecessary recharges.": "అవసరమయ్యే రీచార్జీలు నివారించేందుకు డేటా వినియోగాన్ని గమనించండి.",
        "🏥 Health Tip": "🏥 ఆరోగ్య సూచన",
        "Maintain a small emergency health fund.": "ఒక చిన్న అత్యవసర ఆరోగ్య నిధిని ఉంచండి.",
        "💡 Electricity Tip": "💡 విద్యుత్ సూచన",
        "Unplug unused devices to reduce electricity costs.": "విద్యుత్ ఖర్చును తగ్గించేందుకు వాడని పరికరాలను అన్‌ప్లగ్ చేయండి."
    }
}

tip_title = localized_tips.get(langs[language], {}).get(default_title, default_title)
tip_text = localized_tips.get(langs[language], {}).get(default_tip, default_tip)
st.info(f"{tip_title}: {tip_text}")

# ───────────── Input Section ─────────────
with st.form("input_form"):
    expenses_text = st.text_area("📝 Enter expenses (e.g. 'Spent 300 on groceries, pay rent 5000'):")
    salary = st.number_input("💰 Monthly Salary (₹):", min_value=0.0, step=100.0, format="%.2f")
    goal = st.text_input("🎯 Goal (e.g., 'Buy a car')")
    goal_amount = st.number_input("🎯 Goal Amount (₹):", min_value=1000.0, step=500.0)
    goal_deadline = st.date_input("📅 Goal Deadline")
    phone_number = st.text_input("📱 Enter Mobile Number for Alerts")
    submitted = st.form_submit_button("Analyze")

if not submitted:
    st.stop()

if not expenses_text.strip():
    st.warning("⚠️ Please enter some expenses to analyze.")
    st.stop()

# ───────────── NLP Categorization Setup ─────────────
try:
    with st.spinner("🔄 Loading NLP model..."):
        nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    st.error(f"❌ Failed to load NLP model: {e}")
    st.stop()

categories = {
    "Food": ["groceries","lunch","dinner","snacks","restaurant","coffee"],
    "Travel": ["bus","train","uber","taxi","fuel","petrol","cab","flight"],
    "Electricity": ["electricity bill","power bill","water bill","energy"],
    "Shopping": ["clothes","shopping","mall","amazon","online purchase"],
    "Mobile & Internet": ["mobile bill","recharge","data pack","internet"],
    "Health": ["hospital","medicine","doctor","checkup"],
    "Education": ["books","tuition","school","college","exam fee"],
    "Entertainment": ["movie","netflix","cinema","game","fun","youtube"],
    "Rent & Housing": ["rent","flat","room","maintenance"],
    "Others": []
}
category_embeddings = { cat: nlp_model.encode(phrases) for cat, phrases in categories.items() if phrases }

def get_category(desc):
    desc_embed = nlp_model.encode(desc)
    best_cat, best_score = "Others", 0.4
    for cat, embeds in category_embeddings.items():
        sim = util.cos_sim(desc_embed, embeds).max().item()
        if sim > best_score:
            best_cat, best_score = cat, sim
    return best_cat

# ───────────── Transaction Extraction ─────────────
pattern = r'(?P<desc>[a-zA-Z\s]+?)\s*(on|for)?\s*(₹|Rs\.?)?\s?(?P<amt>\d+(\.\d+)?)'
translated_expenses = translate_expenses(expenses_text)
matches = re.findall(pattern, translated_expenses)

if not matches:
    st.error("❌ No valid transactions found.")
    st.stop()

today = datetime.today()
records = []
for idx, match in enumerate(matches):
    desc = match[0].strip()
    amt = float(match[3])
    date = today - timedelta(days=(len(matches) - idx)//2)
    records.append([date.date(), desc, amt])

df = pd.DataFrame(records, columns=["Date","Description","Amount"])
df["Date"] = pd.to_datetime(df["Date"])
df["Category"] = df["Description"].apply(get_category)
st.success("✅ Transactions processed successfully!")

# ───────────── LSTM Forecast ─────────────
daily = df.groupby("Date")["Amount"].sum().reset_index()
scaler = MinMaxScaler()
daily["Scaled"] = scaler.fit_transform(daily[["Amount"]])
seq_len = 3
X, y = [], []
for i in range(len(daily) - seq_len):
    X.append(daily["Scaled"].iloc[i:i+seq_len].values)
    y.append(daily["Scaled"].iloc[i+seq_len])
X = np.array(X).reshape(-1, seq_len, 1)
y = np.array(y)

prediction = None
if len(X) >= 3:
    model = Sequential([
        Input(shape=(seq_len, 1)),
        LSTM(64, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=20, verbose=0)
    last_seq = daily["Scaled"].values[-seq_len:].reshape(1, seq_len, 1)
    pred_scaled = model.predict(last_seq)[0][0]
    prediction = scaler.inverse_transform([[pred_scaled]])[0][0]
else:
    st.warning("📉 Not enough data to make a prediction.")

# ───────────── Budget Comparison ─────────────
budget_rules = {
    "Rent & Housing": 30, "Food": 15, "Travel": 10, "Shopping": 10,
    "Health": 5, "Mobile & Internet": 5, "Electricity": 5,
    "Education": 5, "Entertainment": 5, "Others": 10
}
budget_df = pd.DataFrame([{ "Category": cat, "Allocated": salary * pct / 100 } for cat, pct in budget_rules.items()])
actual_spend = df.groupby("Category")["Amount"].sum().reset_index()
comp_df = pd.merge(budget_df, actual_spend, on="Category", how="left").fillna(0)
comp_df["Difference"] = comp_df["Allocated"] - comp_df["Amount"]
comp_df["Status"] = comp_df["Difference"].apply(lambda x: "✅ Within Budget" if x >= 0 else "⚠️ Overspent")

# ───────────── Visuals ─────────────
st.subheader("📊 Spending Overview")
col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=daily, x="Date", y="Amount", marker="o", ax=ax1)
    if prediction:
        ax1.axhline(prediction, linestyle="--", color="red", label=f"Predicted ₹{prediction:.2f}")
        ax1.legend()
    ax1.set_title("📈 Daily Spending Forecast")
    st.pyplot(fig1)
with col2:
    fig2, ax2 = plt.subplots()
    sns.barplot(data=comp_df, x="Category", y="Allocated", label="Budget", color="lightblue", ax=ax2)
    sns.barplot(data=comp_df, x="Category", y="Amount", label="Spent", color="salmon", ax=ax2)
    plt.xticks(rotation=45)
    ax2.legend()
    ax2.set_title("💸 Budget vs Spent")
    st.pyplot(fig2)

# ───────────── Suggestions & Goal Tracker ─────────────
st.subheader("💡 AI-Based Finance Suggestions")
top3 = df.groupby("Category")["Amount"].sum().nlargest(3)
total_spent = df["Amount"].sum()
remaining = salary - total_spent
suggested_saving = salary * 0.20
actual_saving = max(0, remaining)

st.markdown(f"""
- **Total Salary:** ₹{salary:.2f}  
- **Total Spent:** ₹{total_spent:.2f}  
- **Remaining Balance:** ₹{remaining:.2f}  
- **Suggested Saving (20%):** ₹{suggested_saving:.2f}  
- **Actual Saving:** ₹{actual_saving:.2f}
""")

if actual_saving >= suggested_saving:
    st.success("✅ You're saving well! Follow the 50/30/20 rule.")
else:
    st.warning("⚠️ You're spending more than recommended.")
    st.markdown("**Suggestions to Save Better:**")
    if "Entertainment" in top3.index:
        st.write("- 🎮 Cut back on streaming or gaming.")
    if "Shopping" in top3.index:
        st.write("- 🛍️ Limit shopping or postpone non-essential buys.")
    if "Food" in top3.index:
        st.write("- 🍔 Cook more at home and reduce dining out.")
    if remaining < 0:
        st.error("🚨 You've overspent this month!")

# ───────────── Goal Tracker ─────────────
if goal and goal_amount > 0:
    st.subheader("🎯 Goal Tracker")
    saved_percent = min(actual_saving / goal_amount * 100, 100)
    days_left = (goal_deadline - datetime.today().date()).days
    st.progress(saved_percent / 100, text=f"{goal}: {saved_percent:.1f}% saved, ₹{actual_saving:.2f} / ₹{goal_amount:.2f} (Deadline: {goal_deadline}, {days_left} days left)")

## ───────────── SMS Alert Simulation ─────────────
user_email = st.text_input("📧 Enter Your Email for Alerts")

# Define overspending categories early (so we can use them in the email body)
overspent_cats = comp_df[comp_df["Status"] == "⚠️ Overspent"]["Category"].tolist()

def send_email_alert(to_email, subject, body):
    from_email = "navya.pati75@gmail.com" # Replace with your email address
    app_password = "qtpp hjjj uofj xomh" # Replace with Gmail App Password

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(from_email, app_password)
            server.sendmail(from_email, to_email, msg.as_string())
            return True
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")
        return False

if user_email:
    subject = "📊 Your AI-Based Finance Report"
    body = f"""
Hello,

Here's your monthly finance summary from AI Finance Manager:

🧾 Total Salary: ₹{salary:.2f}
💸 Total Spent: ₹{total_spent:.2f}
💰 Remaining Balance: ₹{remaining:.2f}
📈 Suggested Saving (20%): ₹{suggested_saving:.2f}
✅ Actual Saving: ₹{actual_saving:.2f}

📊 Overspending Categories: {', '.join(overspent_cats) if overspent_cats else 'None'}

🎯 Goal: {goal}
🏁 Target: ₹{goal_amount:.2f} by {goal_deadline}
📊 Progress: {saved_percent:.1f}% saved (₹{actual_saving:.2f})

Stay financially smart!
- AI Finance Manager
    """.strip()

    if send_email_alert(user_email, subject, body):
        st.success(f"📧 Email alert sent successfully to {user_email}")

    # Display Overspending Alerts in UI
    if overspent_cats:
        st.error(f"🚨 Overspending Alert for: {', '.join(overspent_cats)}")
        for cat in overspent_cats:
            if cat == "Shopping":
                st.info("🛍️ Consider delaying non-essential shopping this month.")
            elif cat == "Food":
                st.info("🍔 Cook at home to save more on food expenses.")
            elif cat == "Entertainment":
                st.info("🎮 Cut down on subscriptions or gaming apps.")
            elif cat == "Travel":
                st.info("🚌 Use public transport or reduce fuel use.")

    if actual_saving < suggested_saving:
        st.warning(f"📉 You're below recommended savings. Try to save ₹{suggested_saving:.0f} per month.")
    else:
        st.success("✅ Good job! Your savings are on track this month.")


# ───────────── Summary Download ─────────────
st.download_button(
    "📥 Download Monthly Summary",
    data=pd.DataFrame([{
        "Salary": salary,
        "Spent": total_spent,
        "Remaining": remaining,
        "Suggested Savings": suggested_saving,
        "Actual Savings": actual_saving,
        "Goal": goal,
        "Goal Amount": goal_amount,
        "Goal Deadline": goal_deadline,
        "Saved %": f"{saved_percent:.1f}%" if goal else "-"
    }]).to_csv(index=False),
    file_name="monthly_summary.csv"
)
# ───── Advanced Features Block ─────
st.subheader("🎛️ Advanced Settings & Insights")
# 🎨 Dark Mixed Color Themes Only
theme = st.selectbox("🎨 Choose Dark Theme:", [
    "Default",
    "Midnight Ocean",
    "Royal Noir",
    "Obsidian Teal",
    "Purple Haze",
    "Charcoal Nebula"  # ✅ New theme
])

if theme != "Default":
    theme_styles = {
        "Midnight Ocean": """
            body, .stApp {
                background: linear-gradient(to right, #000428, #004e92);
                color: #E0F7FA;
            }
        """,
        "Royal Noir": """
            body, .stApp {
                background-color: #1A1A2E;
                color: #F0E68C;
            }
        """,
        "Obsidian Teal": """
            body, .stApp {
                background: linear-gradient(to right, #0f0c29, #302b63, #24243e);
                color: #A0F0ED;
            }
        """,
        "Purple Haze": """
            body, .stApp {
                background: linear-gradient(to right, #360033, #0b8793);
                color: #E1BEE7;
            }
        """,
        "Charcoal Nebula": """
            body, .stApp {
                background: linear-gradient(to right, #2B2B2B, #3B3B5F, #1C1C3A);
                color: #D1D1E9;
            }
        """  # ✅ New style
    }

    # Inject selected theme style
    st.markdown(f"<style>{theme_styles[theme]}</style>", unsafe_allow_html=True)

# Optional: Add an example header to see the style effect
st.markdown(f"## 🌌 You selected: {theme}")

# 2. 👥 Multi-User Profiles
st.markdown("### 👤 Switch Profile")
users = ["Default"] + list(df["User"].unique()) if "User" in df.columns else ["Default"]
selected_user = st.selectbox("Choose User:", users)
filtered_df = df[df["User"] == selected_user] if selected_user != "Default" and "User" in df.columns else df
st.success(f"Currently viewing: **{selected_user}**")

# 3. 📅 Smart Calendar View
st.markdown("### 🗓️ Calendar View")

# Ensure date column is in datetime format
filtered_df["Date"] = pd.to_datetime(filtered_df["Date"], errors='coerce')

# Aggregate spendings by date
calendar_data = filtered_df.groupby("Date")["Amount"].sum().reset_index()

# Add mood emoji based on spending amount
calendar_data["Mood"] = calendar_data["Amount"].apply(lambda x: "🤑" if x > 1000 else "🙂")

# Only show chart if data is available
if not calendar_data.empty:
    import plotly.express as px
    fig = px.scatter(calendar_data, x="Date", y="Amount", text="Mood", title="Spending Calendar")
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available for calendar view.")

# 4. 🧠 AI Emotion-Based Financial Tips
from transformers import pipeline
st.markdown("### 💬 How are you feeling today?")

@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_sentiment_pipeline()

mood_input = st.text_input("Type your emotion or mood (e.g., happy, stressed):")

if mood_input:
    emotion = sentiment_pipeline(mood_input)[0]
    label = emotion["label"]
    score = emotion["score"]

    if label == "POSITIVE":
        st.success("You're feeling good! Great time to invest or plan for goals 🚀")
    elif label == "NEGATIVE":
        st.warning("Low mood detected — try saving a small amount for a win 💡")
    else:
        st.info("Stay consistent with tracking to boost control 📊")

# ───── ✨ Animated Transitions with Framer Motion ─────
import streamlit.components.v1 as components

st.markdown("### ✨ Animated Section")
animation_code = """
<div id="fade" style="opacity: 0; transition: opacity 2s ease-in-out;">
  <h3 style="text-align: center; color: teal;">🌟 Welcome to Your Smart Finance Hub 🌟</h3>
  <p style="text-align: center;">Experience smooth transitions, insights, and control at your fingertips!</p>
</div>
<script>
  const el = document.getElementById("fade");
  window.onload = () => { el.style.opacity = 1; };
</script>
"""
components.html(animation_code, height=200)

# ───── 🧠 AI Monthly Budget Adjuster ─────
def ai_budget_adjuster(expenses_df):
    st.markdown("### 🧠 AI Monthly Budget Adjuster")
    category_group = expenses_df.groupby("Category")["Amount"].sum()
    avg_spending = category_group.mean()

    adjusted_budget = {
        cat: round(amount * 0.9 if amount > avg_spending else amount * 1.1, 2)
        for cat, amount in category_group.items()
    }

    st.info("🔄 Adjusted Budget Suggestions Based on Your Expenses:")
    for cat, val in adjusted_budget.items():
        st.markdown(f"- **{cat}**: ₹{val:.2f}")

# ───── 📥 AI-Based Tax Deduction Estimator ─────
def tax_deduction_estimator(expenses_df):
    st.markdown("### 📥 AI-Based Tax Deduction Estimator")
    deduction_categories = {
        "Health": 25000,
        "Education": 150000,
        "Rent": 60000,
        "Insurance": 50000,
    }

    total_deduction = 0
    for cat, limit in deduction_categories.items():
        if cat in expenses_df["Category"].unique():
            amount = expenses_df[expenses_df["Category"] == cat]["Amount"].sum()
            deduction = min(amount, limit)
            st.markdown(f"- **{cat}**: ₹{deduction} (Max Allowed: ₹{limit})")
            total_deduction += deduction

    st.success(f"🧾 Estimated Total Tax Deductible: ₹{total_deduction}")


def generate_summary_pdf(data):

    # 1. Generate your matplotlib chart and save to a temporary file
    fig, ax = plt.subplots()
    data.columns = data.columns.str.strip().str.lower()  # Normalize column names
    ax.bar(data["category"], data["amount"], color="skyblue")
    ax.set_title("Spending by Category - August")
    ax.set_ylabel("Amount (₹)")
    plt.xticks(rotation=45)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        fig.tight_layout()
        fig.savefig(tmpfile.name)
        chart_path = tmpfile.name

    # 2. Create PDF and use the temp image path
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="month Spending Summary", ln=True, align='C')
    pdf.image(chart_path, x=10, y=30, w=180)

    # 3. Save the final PDF
    pdf.output("summary.pdf")


# ───── 📊 AI Financial Insights Demo ─────
st.markdown("---")
if st.checkbox("📊 Show AI Financial Insights Demo"):
    demo_data = pd.DataFrame({
        "Category": ["Rent", "Food", "Health", "Education", "Insurance", "Entertainment", "Health", "Rent"],
        "Amount": [12000, 5000, 8000, 10000, 4000, 3000, 2000, 12000]
    })

    st.markdown("### 📁 Auto-Generated Demo Data")
    st.dataframe(demo_data)

    ai_budget_adjuster(demo_data)
    tax_deduction_estimator(demo_data)

    generate_summary_pdf(demo_data)
