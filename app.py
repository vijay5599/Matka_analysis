import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import textwrap
from datetime import datetime, timedelta
import altair as alt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from scraper import scrape_mahadevi_chart
from models import (
    load_data, MatkaPredictionEngine, get_basic_statistics, CUT_NUMBERS, get_panas_for_digit,
    predict_date_touch, predict_yesterday_sum, predict_weekly_repeat, predict_panel_multiplier,
    predict_custom_formula, predict_ai_pattern, predict_adaptive_line, run_formula_backtest,
    find_matching_formulas, evaluate_all_models_for_date, find_best_custom_formula
)
import google.generativeai as genai

def generate_ai_report(prediction, history_summary, model_weights, kelly_stakes, api_key, model_name):
    """
    Calls the Google Gemini API to generate an analytical Satta Matka report
    for educational purposes.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=(
                "You are Antigravity, a professional AI Satta Matka analyst and risk mitigation advisor. "
                "Your role is to analyze predictions, probability spreads, and historical trends for educational and research purposes. "
                "Format your response in beautiful, premium GitHub-flavored markdown with clean spacing. "
                "Use bullet points, clear tables, and warning/tip alerts (e.g. > [!NOTE], > [!WARNING]). "
                "Maintain a professional, analytical, yet cautious tone. Emphasize that this is for educational/simulated research only. "
                "Avoid mentioning real money bets without warning that it's educational only."
            )
        )
        
        prompt = f"""
        Analyze the following Satta Matka prediction state:
        
        --- TARGET DRAW ---
        Target Weekday: {prediction['target_weekday']}
        Predicted Jodi: {prediction['predicted_jodi']}
        Predicted Open Digit: {prediction['open_digit']}
        Predicted Close Digit: {prediction['close_digit']}
        Overall Confidence: {prediction['overall_confidence']:.2%}
        Open Single Confidence: {prediction['open_confidence']:.2%}
        Close Single Confidence: {prediction['close_confidence']:.2%}
        
        --- ALTERNATIVE PANAS ---
        Top Open Panas: {prediction['top3_open_panas']}
        Top Close Panas: {prediction['top3_close_panas']}
        
        --- ENSEMBLE WEIGHTS ---
        Markov Chain Model Weight: {model_weights.get('markov', 0):.1%}
        Pattern Matcher Weight: {model_weights.get('pattern', 0):.1%}
        Frequency Model Weight: {model_weights.get('freq', 0):.1%}
        Random Forest Model Weight: {model_weights.get('ml', 0):.1%}
        Gradient Boosting Model Weight: {model_weights.get('gbm', 0):.1%}
        
        --- OPTIMAL BET KELLY STAKES ---
        * Open Bet Stake Recommendation: {kelly_stakes.get('open', '0%')}
        * Open Cut Bet Stake Recommendation: {kelly_stakes.get('open_cut', '0%')}
        * Close Bet Stake Recommendation: {kelly_stakes.get('close', '0%')}
        * Close Cut Bet Stake Recommendation: {kelly_stakes.get('close_cut', '0%')}
        * Jodi Bet Stake Recommendation: {kelly_stakes.get('jodi', '0%')}
        * Jodi Cut Bet Stake Recommendation: {kelly_stakes.get('jodi_cut', '0%')}
        
        --- HISTORY STATS (LAST 10 DRAWS) ---
        {history_summary}
        
        Please generate an AI Analyst Report including:
        1. 🔮 **Prediction Analysis**: Evaluate the strength of the predicted Jodi ({prediction['predicted_jodi']}) and single digits. Discuss the influence of the models based on the active weights.
        2. 🔢 **Cut Number and Hedge Strategy**: Analyze the safety of using the Cut numbers (Open Cut: {(prediction['open_digit']+5)%10}, Close Cut: {(prediction['close_digit']+5)%10}) as hedges.
        3. 💰 **Kelly Allocator Review**: Critique the Kelly Criterion stake suggestions. Discuss how risk-averse researchers should scale their fractional stakes (e.g. Half vs. Quarter Kelly) given the volatility.
        4. ⚠️ **Educational Advisory**: Provide a warning highlighting why Satta Matka is a random sequence generator and why mathematical models are only for academic/simulation studies.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ **Error generating AI analysis:** {str(e)}"

def get_history_summary(df):
    """Summarizes the last 10 draws for LLM context."""
    recent = df.tail(10).iloc[::-1]
    summary_lines = []
    for _, row in recent.iterrows():
        summary_lines.append(
            f"- Date: {row['date'].strftime('%Y-%m-%d')} ({row['weekday']}) | Jodi: {row['jodi']} | Open: {row['open_single']} (Pana: {row['open_pana']}) | Close: {row['close_single']} (Pana: {row['close_pana']})"
        )
    return "\n".join(summary_lines)

# Page config
st.set_page_config(
    page_title="AI Satta Matka Predictor & Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "prediction_model_selectbox" not in st.session_state:
    st.session_state.prediction_model_selectbox = "Machine Learning Ensemble"
if "custom_formula_text_input" not in st.session_state:
    st.session_state.custom_formula_text_input = "(Open + Close) * 2"


def apply_formula_callback(expr):
    st.session_state.prediction_model_selectbox = "Custom User Formula"
    st.session_state.custom_formula_text_input = expr
    st.toast(f"Applied formula: {expr}", icon="🔥")


def render_html(html_content: str):
    """Render HTML content by stripping leading/trailing whitespace from each line
    to prevent Markdown parser from interpreting indented lines as code blocks.
    """
    clean_lines = [line.strip() for line in html_content.splitlines()]
    clean_html = "\n".join(clean_lines)
    st.markdown(clean_html, unsafe_allow_html=True)

# Custom Premium Vanilla CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Global Typography Override */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0b0e14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* App Title Gradient styling */
    .title-gradient {
        background: linear-gradient(135deg, #ff8c00 0%, #ff0055 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
        text-shadow: 0 0 40px rgba(255, 140, 0, 0.1);
    }
    
    .subtitle-text {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Custom Card container */
    div[data-testid="stVerticalBlockBorder"], .dashboard-card {
        background: rgba(18, 24, 38, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        margin-bottom: 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="stVerticalBlockBorder"]:hover, .dashboard-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 140, 0, 0.25) !important;
        box-shadow: 0 15px 40px rgba(255, 140, 0, 0.08) !important;
    }
    
    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Glowing Predictions Badge */
    .prediction-value-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        margin: 20px 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 2px solid rgba(255, 140, 0, 0.3);
        border-radius: 18px;
        padding: 16px 28px;
        text-align: center;
        min-width: 140px;
        box-shadow: 0 8px 24px rgba(255, 140, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #ff8c00, #ff0055);
    }
    
    .prediction-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 1.5px;
        margin-bottom: 4px;
        font-weight: 500;
    }
    
    .prediction-digit {
        font-size: 3rem;
        font-weight: 800;
        color: #ff8c00;
        text-shadow: 0 0 15px rgba(255, 140, 0, 0.3);
        line-height: 1;
    }
    
    .prediction-pana {
        font-size: 1.1rem;
        font-family: 'Space Grotesk', sans-serif;
        color: #e2e8f0;
        margin-top: 6px;
        letter-spacing: 2px;
        font-weight: 600;
    }
    
    .jodi-box {
        background: linear-gradient(135deg, rgba(255, 140, 0, 0.15) 0%, rgba(255, 0, 85, 0.15) 100%);
        border: 2px solid rgba(255, 0, 85, 0.4);
        border-radius: 18px;
        padding: 20px 32px;
        text-align: center;
        min-width: 160px;
        box-shadow: 0 8px 30px rgba(255, 0, 85, 0.15);
        position: relative;
    }
    
    .jodi-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #ff8c00, #ff0055);
    }
    
    .jodi-digit {
        font-size: 3.8rem;
        font-weight: 800;
        color: #ff0055;
        text-shadow: 0 0 20px rgba(255, 0, 85, 0.4);
        line-height: 1;
    }
    
    /* Custom Badge */
    .custom-badge {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 4px 8px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #cbd5e1;
    }
    
    .custom-badge-orange {
        background: rgba(255, 140, 0, 0.1);
        border: 1px solid rgba(255, 140, 0, 0.3);
        border-radius: 8px;
        padding: 4px 8px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #ffa53b;
    }
    
    /* Heatmap styling */
    .heatmap-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        color: #e2e8f0;
    }
    
    .heatmap-table th {
        padding: 10px;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        border-bottom: 2px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        color: #94a3b8;
    }
    
    .heatmap-table td {
        padding: 10px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.02);
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    .heatmap-row-label {
        font-weight: 600 !important;
        color: #ff8c00;
        background: rgba(255, 140, 0, 0.03);
        border-right: 2px solid rgba(255, 140, 0, 0.1) !important;
    }
    
    /* Stats box grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 16px;
    }
    
    .stat-mini-box {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }
    
    .stat-mini-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ff8c00;
        margin-bottom: 2px;
    }
    
    .stat-mini-lbl {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* --- CUSTOM PREMIUM LOADING ANIMATION & OPACITY OVERRIDES --- */
    
    /* Prevent the default grey-out fading of elements during reruns */
    [data-stale="true"], 
    div[data-testid="stAppViewContainer"] [data-stale="true"],
    [data-testid="stForm"] [data-stale="true"] {
        opacity: 1 !important;
        filter: none !important;
    }
    
    /* 1. Top-edge glowing loading line animation */
    div[data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #ff8c00 0%, #ff0055 50%, #7a00ff 100%);
        background-size: 200% 100%;
        z-index: 1000001; /* Higher than the blocker backdrop */
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.3s ease-in-out;
    }
    
    /* When Streamlit status widget is active, display the top progress bar and slide the gradient */
    body:has([data-testid="stStatusWidget"]) div[data-testid="stAppViewContainer"]::before {
        opacity: 1;
        animation: premium-loader-flow 1.5s infinite linear;
    }
    
    @keyframes premium-loader-flow {
        0% {
            background-position: 0% 50%;
        }
        100% {
            background-position: -200% 50%;
        }
    }
    
    /* 2. Full-screen Glassmorphic Blocker Backdrop */
    body::after {
        content: "Analyzing Live Models & Risk Matrices... 🔮";
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(11, 14, 20, 0.7); /* Deep premium dark overlay */
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        z-index: 999999;
        display: flex;
        justify-content: center;
        align-items: center;
        padding-top: 120px; /* Shifts text down so it sits below the spinner */
        color: #e2e8f0;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        letter-spacing: 1px;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Block pointer events (clicks/scrolls) and fade in the backdrop when running */
    body:has([data-testid="stStatusWidget"])::after {
        opacity: 1;
        pointer-events: auto;
    }
    
    /* 3. Glowing Central Spinner */
    body::before {
        content: "";
        position: fixed;
        top: calc(50% - 30px);
        left: calc(50% - 30px);
        width: 60px;
        height: 60px;
        border: 4px solid rgba(255, 140, 0, 0.15);
        border-top: 4px solid #ff8c00;
        border-right: 4px solid #ff0055;
        border-radius: 50%;
        z-index: 1000000; /* Above the backdrop overlay */
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Fade in and animate spinner rotation when running */
    body:has([data-testid="stStatusWidget"])::before {
        opacity: 1;
        animation: premium-spinner-spin 1s infinite linear;
    }
    
    @keyframes premium-spinner-spin {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }
    
</style>
""", unsafe_allow_html=True)

# Helper function to trigger scraping/updating data
def sync_live_data():
    with st.spinner("Scraping live records from tara567... Please wait."):
        try:
            records = scrape_mahadevi_chart()
            output_path = "/Users/vijay5599/Developer/Projects/AI Agents/Matka_analysis/mahadevi_history.json"
            with open(output_path, "w") as f:
                json.dump(records, f, indent=4)
            st.toast("Database updated successfully!", icon="🔥")
            # Force cache clear for reload
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Failed to scrape: {e}")

# Load cached data
@st.cache_data
def get_cached_dataset():
    df_raw, df_valid = load_data()
    stats = get_basic_statistics(df_valid)
    return df_raw, df_valid, stats

@st.cache_data
def run_selected_method_backtest(df_valid, method_name, custom_formula="", limit=30):
    if method_name == "ml_ensemble":
        default_w = {"freq": 0.10, "markov": 0.20, "pattern": 0.20, "ml": 0.25, "gbm": 0.25}
        res = engine.backtest(test_draws_count=limit, weights=default_w)
        return {
            "touchRate": f"{(res['top3_rate_open'] + res['top3_rate_close'])/2 * 100:.1f}",
            "singleRate": f"{res['accuracy_open'] * 100:.1f}",
            "jodiRate": f"{res['accuracy_jodi'] * 100:.1f}",
            "panaRate": "0.0",
            "accuracy_open": res["accuracy_open"],
            "accuracy_close": res["accuracy_close"],
            "top3_rate_open": res["top3_rate_open"],
            "top3_rate_close": res["top3_rate_close"],
            "accuracy_jodi": res["accuracy_jodi"]
        }
    else:
        res = run_formula_backtest(df_valid, method_name, custom_formula, limit)
        if res:
            return {
                "touchRate": res["touchRate"],
                "singleRate": res["singleRate"],
                "jodiRate": res["jodiRate"],
                "panaRate": res["panaRate"],
                "accuracy_open": float(res["singleRate"])/100.0,
                "accuracy_close": float(res["singleRate"])/100.0,
                "top3_rate_open": float(res["touchRate"])/100.0,
                "top3_rate_close": float(res["touchRate"])/100.0,
                "accuracy_jodi": float(res["jodiRate"])/100.0
            }
        else:
            return {
                "touchRate": "0.0", "singleRate": "0.0", "jodiRate": "0.0", "panaRate": "0.0",
                "accuracy_open": 0.0, "accuracy_close": 0.0,
                "top3_rate_open": 0.0, "top3_rate_close": 0.0, "accuracy_jodi": 0.0
            }

def make_compatible_prediction(algebraic_res, target_date, known_open=None):
    if "open_probs" in algebraic_res:
        pred = algebraic_res.copy()
        if known_open is not None:
            pred["open_digit"] = known_open
            o_probs = np.zeros(10)
            o_probs[known_open] = 1.0
            pred["open_probs"] = o_probs.tolist()
            pred["open_confidence"] = 1.0
            pred["predicted_jodi"] = f"{known_open}{pred['close_digit']}"
            pred["top3_open_digits"] = [known_open] + [d for d in pred["top3_open_digits"] if d != known_open][:2]
            pred["top3_open_panas"] = get_panas_for_digit(known_open)[:3]
            pred["overall_confidence"] = pred["close_confidence"]
        return pred

    touch_digits = algebraic_res.get("touchDigits", [])
    jodis = algebraic_res.get("jodis", [])
    panas = algebraic_res.get("panas", [])

    if not touch_digits:
        touch_digits = ["0", "5", "1", "6"]
    if not jodis:
        jodis = ["05", "50", "16", "61"]
    if not panas:
        panas = ["122", "127", "113", "118"]

    if known_open is not None:
        open_digit = known_open
    else:
        open_digit = int(touch_digits[0])

    close_digit = int(touch_digits[1]) if len(touch_digits) > 1 else int(touch_digits[0])

    open_probs = np.ones(10) * 0.05
    close_probs = np.ones(10) * 0.05
    for i, digit_str in enumerate(touch_digits):
        try:
            d = int(digit_str)
            weight = 0.40 - i * 0.10
            if weight > 0:
                open_probs[d] = max(open_probs[d], weight)
                close_probs[d] = max(close_probs[d], weight)
        except:
            pass

    if known_open is not None:
        open_probs = np.zeros(10)
        open_probs[known_open] = 1.0

    open_probs = open_probs / open_probs.sum()
    close_probs = close_probs / close_probs.sum()

    open_panas = []
    close_panas = []
    for p in panas:
        try:
            p_sum = sum(int(digit) for digit in p) % 10
            if p_sum == open_digit:
                open_panas.append(p)
            elif p_sum == close_digit:
                close_panas.append(p)
        except:
            pass

    open_panas.extend([p for p in get_panas_for_digit(open_digit) if p not in open_panas])
    close_panas.extend([p for p in get_panas_for_digit(close_digit) if p not in close_panas])

    top3_open_panas = open_panas[:3]
    top3_close_panas = close_panas[:3]

    top3_open_digits = [int(x) for x in touch_digits[:3]]
    if known_open is not None:
        top3_open_digits = [known_open] + [d for d in top3_open_digits if d != known_open][:2]
    top3_close_digits = [int(x) for x in touch_digits[1:4]]
    if not top3_close_digits:
        top3_close_digits = [close_digit]

    predicted_jodi = f"{open_digit}{close_digit}"

    return {
        "target_weekday": target_date.strftime("%A"),
        "open_digit": open_digit,
        "open_confidence": float(open_probs[open_digit]),
        "open_probs": open_probs.tolist(),
        "top3_open_digits": top3_open_digits,
        "top3_open_panas": top3_open_panas,
        "close_digit": close_digit,
        "close_confidence": float(close_probs[close_digit]),
        "close_probs": close_probs.tolist(),
        "top3_close_digits": top3_close_digits,
        "top3_close_panas": top3_close_panas,
        "predicted_jodi": predicted_jodi,
        "overall_confidence": float(open_probs[open_digit] * close_probs[close_digit]),
        "name": algebraic_res.get("name", "Algebraic Model"),
        "description": algebraic_res.get("description", ""),
        "touchDigits": touch_digits,
        "jodis": jodis,
        "panas": panas
    }

def highlight_hits(val):
    if val is True:
        return 'background-color: rgba(34, 197, 94, 0.15); color: #22c55e;'
    elif val is False:
        return 'background-color: rgba(239, 68, 68, 0.05); color: #ef4444;'
    return ''

def get_all_model_predictions_for_future(df_history, target_date):
    models = [
        {"key": "ml_ensemble", "name": "Machine Learning Ensemble"},
        {"key": "date_touch", "name": "Date-wise Touch Scheme"},
        {"key": "yesterday_sum", "name": "Yesterday's Open-Close Sum"},
        {"key": "weekly_repeat", "name": "Weekly Repeat Pattern"},
        {"key": "panel_multiplier", "name": "Panel Digit Multiplier"},
        {"key": "ai_pattern", "name": "AI Neural Pattern Recognizer"},
        {"key": "adaptive_line", "name": "Adaptive Running Line Optimizer"}
    ]
    results = []
    for model in models:
        pred = None
        if model["key"] == "ml_ensemble":
            try:
                pred = engine.predict_next(df_train=df_history, target_weekday=target_date.weekday())
                pred["touchDigits"] = [str(x) for x in pred["top3_open_digits"][:4]]
                while len(pred["touchDigits"]) < 4:
                    pred["touchDigits"].append("0")
                pred["jodis"] = [pred["predicted_jodi"]] + generate_jodis(pred["touchDigits"])[:7]
                pred["panas"] = pred["top3_open_panas"] + pred["top3_close_panas"]
            except Exception as e:
                pass
        elif model["key"] == "date_touch":
            pred = predict_date_touch(target_date)
        elif model["key"] == "yesterday_sum":
            pred = predict_yesterday_sum(df_history, target_date)
        elif model["key"] == "weekly_repeat":
            pred = predict_weekly_repeat(df_history, target_date)
        elif model["key"] == "panel_multiplier":
            pred = predict_panel_multiplier(df_history, target_date)
        elif model["key"] == "ai_pattern":
            pred = predict_ai_pattern(df_history, target_date)
        elif model["key"] == "adaptive_line":
            pred = predict_adaptive_line(df_history, target_date)
        
        if pred:
            results.append({
                **model,
                "touchDigits": pred.get("touchDigits", []),
                "jodis": pred.get("jodis", []),
                "panas": pred.get("panas", [])
            })
    return results

def render_prediction_card(pred, is_past_date=False, actual_row=None):
    name = pred.get("name", "Model Prediction")
    desc = pred.get("description", "")
    touch_digits = pred.get("touchDigits", [str(pred["open_digit"]), str(pred["close_digit"])])
    jodis_list = pred.get("jodis", [pred["predicted_jodi"]])
    panas_list = pred.get("panas", pred["top3_open_panas"] + pred["top3_close_panas"])
    
    act_open_s = str(actual_row["open_single"]) if is_past_date and actual_row is not None else None
    act_close_s = str(actual_row["close_single"]) if is_past_date and actual_row is not None else None
    act_jodi = actual_row["jodi"] if is_past_date and actual_row is not None else None
    act_open_p = actual_row["open_pana"] if is_past_date and actual_row is not None else None
    act_close_p = actual_row["close_pana"] if is_past_date and actual_row is not None else None

    touch_html = ""
    for digit in touch_digits:
        is_hit = is_past_date and (digit == act_open_s or digit == act_close_s)
        bg = "linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.4) 100%)" if is_hit else "rgba(255, 255, 255, 0.03)"
        border = "2px solid #22c55e" if is_hit else "1px solid rgba(255, 255, 255, 0.1)"
        shadow = "box-shadow: 0 0 15px rgba(34, 197, 94, 0.4);" if is_hit else ""
        color = "#22c55e" if is_hit else "#fff"
        touch_html += f"""
        <div style="width: 50px; height: 50px; border-radius: 50%; background: {bg}; border: {border}; display: flex; justify-content: center; align-items: center; font-size: 1.5rem; font-weight: 700; color: {color}; {shadow}">
            {digit}
        </div>
        """
        
    jodis_html = ""
    for jodi in jodis_list:
        is_hit = is_past_date and (jodi == act_jodi)
        bg = "linear-gradient(135deg, rgba(34, 197, 94, 0.25) 0%, rgba(34, 197, 94, 0.4) 100%)" if is_hit else "rgba(255, 255, 255, 0.02)"
        border = "2px solid #22c55e" if is_hit else "1px solid rgba(255, 255, 255, 0.06)"
        color = "#22c55e" if is_hit else "#ff8c00"
        shadow = "box-shadow: 0 0 15px rgba(34, 197, 94, 0.3);" if is_hit else ""
        jodis_html += f"""
        <div style="background: {bg}; border: {border}; color: {color}; font-weight: 700; padding: 6px 12px; border-radius: 8px; font-family: monospace; font-size: 1.1rem; text-align: center; {shadow}">
            {jodi}
        </div>
        """

    panas_html = ""
    for pana in panas_list:
        is_hit = is_past_date and (pana == act_open_p or pana == act_close_p)
        bg = "linear-gradient(135deg, rgba(34, 197, 94, 0.25) 0%, rgba(34, 197, 94, 0.4) 100%)" if is_hit else "rgba(255, 255, 255, 0.02)"
        border = "2px solid #22c55e" if is_hit else "1px solid rgba(255, 255, 255, 0.06)"
        color = "#22c55e" if is_hit else "#e2e8f0"
        shadow = "box-shadow: 0 0 15px rgba(34, 197, 94, 0.3);" if is_hit else ""
        panas_html += f"""
        <div style="background: {bg}; border: {border}; color: {color}; font-weight: 600; padding: 6px 12px; border-radius: 8px; font-family: monospace; font-size: 1rem; text-align: center; {shadow}">
            {pana}
        </div>
        """

    desc_section = f'<p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5; margin-bottom: 20px; background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.04); padding: 12px; border-radius: 10px;"><b>Formula Details:</b> {desc}</p>' if desc else ""

    render_html(f"""
        <div style="margin-top: 25px;">
            <div style="font-size: 0.85rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 1px; margin-bottom: 10px; font-weight: 500;">🔮 OTC Touch Digits (Top 4)</div>
            <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                {touch_html}
            </div>
        </div>
        
        <div style="margin-top: 25px; margin-bottom: 24px; display: grid; grid-template-columns: 1fr; gap: 20px;">
            <div>
                <div style="font-size: 0.85rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 1px; margin-bottom: 10px; font-weight: 500;">🎲 Suggested Jodis (Top 8)</div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                    {jodis_html}
                </div>
            </div>
            <div>
                <div style="font-size: 0.85rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 1px; margin-bottom: 10px; font-weight: 500;">📋 Suggested Panas (Panels)</div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                    {panas_html}
                </div>
            </div>
        </div>
    """)

# Initialize app state or load data
try:
    df_raw, df_valid, hist_stats = get_cached_dataset()
except Exception as e:
    st.warning("Historical data cache not found. Attempting initial scrape...")
    try:
        records = scrape_mahadevi_chart()
        output_path = "/Users/vijay5599/Developer/Projects/AI Agents/Matka_analysis/mahadevi_history.json"
        with open(output_path, "w") as f:
            json.dump(records, f, indent=4)
        df_raw, df_valid, hist_stats = get_cached_dataset()
    except Exception as scrape_err:
        st.error(f"Initialization error: {scrape_err}")
        st.stop()

# Prediction engine setup
engine = MatkaPredictionEngine(df_valid)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 20px;'><span style='font-size: 3rem;'>🔮</span></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-family: Space Grotesk, sans-serif; color: #fff;'>Control Center</h2>", unsafe_allow_html=True)
    
    # Navigation menu
    nav_selection = st.radio(
        "Navigation",
        [
            "🔮 Live Predictions",
            "📊 Charts & Analytics",
            "⚙️ Model Tuning & Backtest",
            "🗂️ Historical Data Explorer"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("<h4 style='font-family: Space Grotesk, sans-serif; color: #ff8c00; margin-bottom: 10px;'>🔮 Prediction Setup</h4>", unsafe_allow_html=True)
    
    dates_list = df_valid["date"].sort_values(ascending=False).dt.strftime("%Y-%m-%d").tolist()
    next_date_str = (df_valid["date"].max() + timedelta(days=1)).strftime("%Y-%m-%d")
    options = [f"Next Draw ({next_date_str})"] + dates_list
    selected_date_choice = st.selectbox("Target Date", options, index=1, help="Select 'Next Draw' to predict tomorrow, or select a past date to backtest and check hits.")
    
    if selected_date_choice.startswith("Next Draw"):
        target_date = df_valid["date"].max() + timedelta(days=1)
    else:
        target_date = datetime.strptime(selected_date_choice, "%Y-%m-%d")
        
    method_map = {
        "Machine Learning Ensemble": "ml_ensemble",
        "Date-wise Touch Scheme": "date_touch",
        "Yesterday's Open-Close Sum": "yesterday_sum",
        "Weekly Repeat Pattern": "weekly_repeat",
        "Panel Digit Multiplier": "panel_multiplier",
        "Custom User Formula": "custom",
        "AI Neural Pattern Recognizer": "ai_pattern",
        "Adaptive Running Line Optimizer": "adaptive_line"
    }
    selected_method_name = st.selectbox("Prediction Model", list(method_map.keys()), key="prediction_model_selectbox")
    selected_method = method_map[selected_method_name]
    
    custom_formula_str = ""
    if selected_method == "custom":
        custom_formula_str = st.text_input(
            "Custom Formula String",
            key="custom_formula_text_input",
            help="Allowed characters: numbers, spaces, operators (+, -, *, /, %, (, ), .). Variables: Open, Close, Jodi, Date, Month."
        )
        
    known_open_input = st.selectbox(
        "Known Open Single",
        ["None"] + [str(i) for i in range(10)],
        index=0,
        help="If the open single digit is already known/drawn, specify it here to refine close and jodi predictions."
    )
    known_open = None if known_open_input == "None" else int(known_open_input)
    
    if selected_method == "ml_ensemble":
        st.markdown("---")
        st.markdown("<h4 style='font-family: Space Grotesk, sans-serif; color: #ff8c00;'>Ensemble Weight Tuning</h4>", unsafe_allow_html=True)
        w_freq = st.slider("Frequency model weight", 0.0, 1.0, 0.10, 0.05)
        w_markov = st.slider("Markov Chain model weight", 0.0, 1.0, 0.20, 0.05)
        w_pattern = st.slider("Pattern Matcher weight", 0.0, 1.0, 0.20, 0.05)
        w_ml = st.slider("Random Forest weight", 0.0, 1.0, 0.25, 0.05)
        w_gbm = st.slider("Gradient Boosting weight", 0.0, 1.0, 0.25, 0.05)
        
        total_w = w_freq + w_markov + w_pattern + w_ml + w_gbm
        if total_w > 0:
            weights = {
                "freq": w_freq / total_w,
                "markov": w_markov / total_w,
                "pattern": w_pattern / total_w,
                "ml": w_ml / total_w,
                "gbm": w_gbm / total_w
            }
        else:
            weights = {"freq": 0.20, "markov": 0.20, "pattern": 0.20, "ml": 0.20, "gbm": 0.20}
            
        with st.sidebar.expander("❓ What are these weights?"):
            st.markdown("""
            The engine combines **5 mathematical and ML models** to generate a consolidated prediction. Tuning these weights changes each model's influence:
            
            * **Frequency Model**: Predicts digits based on their historical occurrence counts (both general and weekday-specific).
            * **Markov Chain**: Evaluates sequential transitions (which digit historically follows the previous draw's digit).
            * **Pattern Matcher**: Analyzes sequence loops, looking back at the last few draws and searching history for identical patterns.
            * **Random Forest**: A bagging ensemble ML model trained on lag sequences, rolling trends, and weekday indicators.
            * **Gradient Boosting (GBM)**: A sequential boosting tree classifier that fits new trees to correct residual prediction errors.
            
            *Weights are normalized automatically to sum to 100%.*
            """)
    else:
        weights = {"freq": 0.10, "markov": 0.20, "pattern": 0.20, "ml": 0.25, "gbm": 0.25}
        
    st.markdown("---")
    
    st.button("🔄 Sync Live Matka Data", on_click=sync_live_data, use_container_width=True)
    
    st.markdown("---")
    st.markdown("<h4 style='font-family: Space Grotesk, sans-serif; color: #a855f7;'>🔮 AI Analyst Setup</h4>", unsafe_allow_html=True)
    
    env_api_key = os.environ.get("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=env_api_key,
        type="password",
        help="Get your API key from Google AI Studio: https://aistudio.google.com/"
    )
    
    ai_model = st.selectbox(
        "AI Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash"],
        index=0
    )
    
    st.markdown("<div style='margin-top: -10px; font-size: 0.8rem;'><a href='https://aistudio.google.com/' target='_blank' style='color: #a855f7; text-decoration: none;'>🔑 Get Free Gemini Key</a></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>"
        f"Last Draw Scraped: {df_valid['date'].max().strftime('%Y-%m-%d')}<br>"
        f"Total Draw History: {len(df_valid)} records"
        f"</div>",
        unsafe_allow_html=True
    )

# ==================== HEADER ====================
st.markdown("<h1 class='title-gradient'>Mahadevi AI Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>An intelligent machine learning, probability-based, and statistical pattern analysis engine built for educational research.</p>", unsafe_allow_html=True)

if nav_selection == "🔮 Live Predictions":
    is_past_date = not selected_date_choice.startswith("Next Draw")

    if is_past_date:
        df_train = df_valid[df_valid["date"] < target_date].copy()
        actual_row = df_valid[df_valid["date"] == target_date].iloc[0]
    else:
        df_train = df_valid.copy()
        actual_row = None

    # Get prediction based on selected method
    if selected_method == "ml_ensemble":
        pred = engine.predict_next(df_train=df_train, target_weekday=target_date.weekday(), weights=weights)
        if known_open is not None:
            pred = make_compatible_prediction(pred, target_date, known_open=known_open)
    else:
        if selected_method == "date_touch":
            raw_pred = predict_date_touch(target_date)
        elif selected_method == "yesterday_sum":
            raw_pred = predict_yesterday_sum(df_train, target_date)
        elif selected_method == "weekly_repeat":
            raw_pred = predict_weekly_repeat(df_train, target_date)
        elif selected_method == "panel_multiplier":
            raw_pred = predict_panel_multiplier(df_train, target_date)
        elif selected_method == "custom":
            raw_pred = predict_custom_formula(df_train, target_date, custom_formula_str)
        elif selected_method == "ai_pattern":
            raw_pred = predict_ai_pattern(df_train, target_date)
        elif selected_method == "adaptive_line":
            raw_pred = predict_adaptive_line(df_train, target_date)
            
        pred = make_compatible_prediction(raw_pred, target_date, known_open=known_open)

    # Check individual hits for styling the main cards
    open_hit_style = ""
    close_hit_style = ""
    jodi_hit_style = ""

    if is_past_date and actual_row is not None:
        act_open = str(actual_row["open_single"])
        act_close = str(actual_row["close_single"])
        act_jodi = str(actual_row["jodi"])
        
        if str(pred['open_digit']) == act_open:
            open_hit_style = "border: 3px solid #22c55e; box-shadow: 0 0 20px rgba(34, 197, 94, 0.4);"
        if str(pred['close_digit']) == act_close:
            close_hit_style = "border: 3px solid #22c55e; box-shadow: 0 0 20px rgba(34, 197, 94, 0.4);"
        if pred['predicted_jodi'] == act_jodi:
            jodi_hit_style = "border: 3px solid #22c55e; box-shadow: 0 0 25px rgba(34, 197, 94, 0.5);"

    # Validation Card
    if is_past_date and actual_row is not None:
        act_open = str(actual_row["open_single"])
        act_close = str(actual_row["close_single"])
        act_jodi = str(actual_row["jodi"])
        act_open_p = str(actual_row["open_pana"])
        act_close_p = str(actual_row["close_pana"])
        
        pred_touch = pred.get("touchDigits", [str(pred["open_digit"]), str(pred["close_digit"])])
        pred_jodis = pred.get("jodis", [pred["predicted_jodi"]])
        pred_panas = pred.get("panas", pred["top3_open_panas"] + pred["top3_close_panas"])
        
        touch_hit = act_open in pred_touch or act_close in pred_touch
        jodi_hit = act_jodi in pred_jodis
        pana_hit = act_open_p in pred_panas or act_close_p in pred_panas
        
        with st.container(border=True):
            st.markdown("<h4 style='color: #22c55e; font-family: Space Grotesk, sans-serif; font-size: 1.25rem; margin-bottom: 15px;'>✅ Past Draw Outcome Verification</h4>", unsafe_allow_html=True)
            
            c_v1, c_v2, c_v3 = st.columns(3)
            with c_v1:
                st.metric("Actual Result", f"{act_open_p} - {act_jodi} - {act_close_p}", help="Format: Open Pana - Jodi - Close Pana")
            with c_v2:
                st.metric("Jodi Match", "✅ HIT" if jodi_hit else "❌ MISS", delta="Success" if jodi_hit else "Failed")
            with c_v3:
                st.metric("Touch Match", "✅ HIT" if touch_hit else "❌ MISS", delta="Success" if touch_hit else "Failed")

    col_main, col_prob = st.columns([1, 1])
    
    with col_main:
        with st.container(border=True):
            render_html(f"""
                <div class="card-title">🔮 Prediction Result ({pred.get('name', 'Selected Model')})</div>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-top: -10px;">
                    Ensemble or algebraic predictions calculated from historical data.
                </p>
                <div class="prediction-value-wrapper">
                    <div class="prediction-box" style="{open_hit_style}">
                        <div class="prediction-label">Open Pana</div>
                        <div class="prediction-digit">{pred['open_digit']}</div>
                        <div class="prediction-pana">{pred['top3_open_panas'][0] if pred['top3_open_panas'] else 'N/A'}</div>
                    </div>
                    <div class="jodi-box" style="{jodi_hit_style}">
                        <div class="prediction-label" style="color: #ff0055;">Predicted Jodi</div>
                        <div class="jodi-digit">{pred['predicted_jodi']}</div>
                        <div style="font-size: 0.8rem; color: #fda4af; margin-top: 8px; font-weight: 600;">CONFIDENCE: {pred['overall_confidence']:.1%}</div>
                    </div>
                    <div class="prediction-box" style="{close_hit_style}">
                        <div class="prediction-label">Close Pana</div>
                        <div class="prediction-digit">{pred['close_digit']}</div>
                        <div class="prediction-pana">{pred['top3_close_panas'][0] if pred['top3_close_panas'] else 'N/A'}</div>
                    </div>
                </div>
            """)
            
            # Display prediction card styling with highlighting
            render_prediction_card(pred, is_past_date=is_past_date, actual_row=actual_row if is_past_date else None)

            # Details table
            st.markdown("<h5 style='color: #cbd5e1; font-family: Space Grotesk, sans-serif; font-size: 1rem; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 4px;'>🎯 Detailed Prediction Sheet</h5>", unsafe_allow_html=True)
            details_data = {
                "Target Weekday": pred['target_weekday'],
                "Predicted Jodi": pred['predicted_jodi'],
                "Jodi Confidence": f"{pred['overall_confidence']:.2%}",
                "Top-3 Open Digits": ", ".join(map(str, pred['top3_open_digits'])),
                "Top-3 Close Digits": ", ".join(map(str, pred['top3_close_digits'])),
                "Open Single Confidence": f"{pred['open_confidence']:.2%}",
                "Close Single Confidence": f"{pred['close_confidence']:.2%}",
                "Alternative Open Panas": ", ".join(pred['top3_open_panas']),
                "Alternative Close Panas": ", ".join(pred['top3_close_panas'])
            }
            details_df = pd.DataFrame(list(details_data.items()), columns=["Metric", "Value"])
            st.dataframe(details_df, use_container_width=True, hide_index=True)
            
            st.markdown(f"""
                <div style="margin-top: 25px; background: rgba(255, 140, 0, 0.05); border: 1px solid rgba(255, 140, 0, 0.15); border-radius: 12px; padding: 15px;">
                    <h6 style="color: #ffa53b; font-family: Space Grotesk, sans-serif; font-size: 0.95rem; margin-bottom: 6px;">🔮 Cut Number Theory (Matka Numerology)</h6>
                    <p style="color:#cbd5e1; font-size:0.85rem; margin:0; line-height: 1.4;">
                        In Satta Matka, the <b>cut number</b> (digit + 5 modulo 10) frequently manifests as a hedge value. 
                        If the prediction fails, its cut version often appears. 
                        Predicted Open Cut is <b>{(pred['open_digit'] + 5) % 10}</b>, and Predicted Close Cut is <b>{(pred['close_digit'] + 5) % 10}</b> (Cut Jodi: <b>{(pred['open_digit'] + 5) % 10}{(pred['close_digit'] + 5) % 10}</b>).
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        # Kelly Stake Optimizer
        with st.container(border=True):
            st.markdown('<div class="card-title">💰 Kelly Criterion Risk & Stake Optimizer</div>', unsafe_allow_html=True)
            st.markdown(
                '<p style="color: #94a3b8; font-size: 0.85rem; margin-top: -10px;">'
                'Calculate optimal mathematical bet sizes based on model probabilities and payout odds.'
                '</p>',
                unsafe_allow_html=True
            )
            
            c_k1, c_k2, c_k3 = st.columns(3)
            with c_k1:
                bankroll = st.number_input("Total Bankroll (pts)", min_value=10, max_value=1000000, value=1000, step=100)
            with c_k2:
                bet_type = st.selectbox("Bet Type / Odds", ["Single Digit (9x)", "Jodi (90x)", "Pana (140x)", "Custom Payout"])
            with c_k3:
                kelly_fraction = st.selectbox("Kelly Fraction", ["Half Kelly (Recommended)", "Full Kelly (Aggressive)", "Quarter Kelly (Conservative)"])
                
            if bet_type == "Single Digit (9x)":
                odds = 9.0
            elif bet_type == "Jodi (90x)":
                odds = 90.0
            elif bet_type == "Pana (140x)":
                odds = 140.0
            else:
                odds = st.number_input("Custom Payout (x)", min_value=1.1, max_value=1000.0, value=9.0, step=0.5)
                
            k_mult = 0.5
            if "Full" in kelly_fraction:
                k_mult = 1.0
            elif "Quarter" in kelly_fraction:
                k_mult = 0.25
                
            b_odds = odds - 1.0
            
            def calc_kelly_stake(prob, b, bank, mult):
                f_star = (prob * (b + 1) - 1.0) / b
                if f_star <= 0:
                    return 0.0, 0.0
                adjusted_f = f_star * mult
                stake = bank * adjusted_f
                return adjusted_f, stake
                
            p_open = pred["open_probs"][pred["open_digit"]]
            o_f, o_stake = calc_kelly_stake(p_open, b_odds, bankroll, k_mult)
            
            open_cut_digit = (pred["open_digit"] + 5) % 10
            p_open_cut = pred["open_probs"][open_cut_digit]
            o_cut_f, o_cut_stake = calc_kelly_stake(p_open_cut, b_odds, bankroll, k_mult)
            
            p_close = pred["close_probs"][pred["close_digit"]]
            c_f, c_stake = calc_kelly_stake(p_close, b_odds, bankroll, k_mult)
            
            close_cut_digit = (pred["close_digit"] + 5) % 10
            p_close_cut = pred["close_probs"][close_cut_digit]
            c_cut_f, c_cut_stake = calc_kelly_stake(p_close_cut, b_odds, bankroll, k_mult)
            
            p_jodi = pred["overall_confidence"]
            j_f, j_stake = calc_kelly_stake(p_jodi, b_odds, bankroll, k_mult)
            
            jodi_cut_str = f"{open_cut_digit}{close_cut_digit}"
            p_jodi_cut = pred["open_probs"][open_cut_digit] * pred["close_probs"][close_cut_digit]
            j_cut_f, j_cut_stake = calc_kelly_stake(p_jodi_cut, b_odds, bankroll, k_mult)
            
            kelly_stakes_dict = {
                "open": f"{o_f:.1%} ({o_stake:.1f} pts)" if o_stake > 0 else "No Bet (0 pts)",
                "open_cut": f"{o_cut_f:.1%} ({o_cut_stake:.1f} pts)" if o_cut_stake > 0 else "No Bet (0 pts)",
                "close": f"{c_f:.1%} ({c_stake:.1f} pts)" if c_stake > 0 else "No Bet (0 pts)",
                "close_cut": f"{c_cut_f:.1%} ({c_cut_stake:.1f} pts)" if c_cut_stake > 0 else "No Bet (0 pts)",
                "jodi": f"{j_f:.2%} ({j_stake:.1f} pts)" if j_stake > 0 else "No Bet (0 pts)",
                "jodi_cut": f"{j_cut_f:.2%} ({j_cut_stake:.1f} pts)" if j_cut_stake > 0 else "No Bet (0 pts)"
            }
            
            if "Single" in bet_type or (bet_type == "Custom Payout" and odds < 20):
                st.markdown("<h5 style='color: #cbd5e1; font-family: Space Grotesk, sans-serif; font-size: 1rem; margin-top: 15px;'>🎯 Recommended Stakes: Single Digits</h5>", unsafe_allow_html=True)
                col_o, col_c = st.columns(2)
                
                with col_o:
                    st.markdown(
                        f"<div style='padding: 15px; background: rgba(255, 140, 0, 0.05); border: 1px solid rgba(255, 140, 0, 0.15); border-radius: 12px; height: 100%;'>"
                        f"<div style='font-size: 0.8rem; color: #ffa53b; text-transform: uppercase; letter-spacing: 1px;'>Open Digit: {pred['open_digit']}</div>"
                        f"<div style='font-size: 1.8rem; font-weight: 700; color: #fff; margin: 5px 0;'>"
                        f"{f'{o_stake:.1f} pts' if o_stake > 0 else 'No Bet (0 pts)'}</div>"
                        f"<div style='font-size: 0.75rem; color: #94a3b8;'>"
                        f"Probability: {p_open:.2%} | Kelly Fraction: {o_f:.1%}"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    if o_cut_stake > 0:
                        st.markdown(
                            f"<div style='padding: 10px; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 10px; margin-top: 10px;'>"
                            f"<span style='font-size: 0.75rem; color: #cbd5e1;'>Hedge Open Cut ({open_cut_digit}): <b>{o_cut_stake:.1f} pts</b> (Prob: {p_open_cut:.1%})</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div style='padding: 10px; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 10px; margin-top: 10px;'>"
                            f"<span style='font-size: 0.75rem; color: #64748b;'>No hedge recommended for Open Cut ({open_cut_digit})</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        
                with col_c:
                    st.markdown(
                        f"<div style='padding: 15px; background: rgba(255, 0, 85, 0.05); border: 1px solid rgba(255, 0, 85, 0.15); border-radius: 12px; height: 100%;'>"
                        f"<div style='font-size: 0.8rem; color: #fda4af; text-transform: uppercase; letter-spacing: 1px;'>Close Digit: {pred['close_digit']}</div>"
                        f"<div style='font-size: 1.8rem; font-weight: 700; color: #fff; margin: 5px 0;'>"
                        f"{f'{c_stake:.1f} pts' if c_stake > 0 else 'No Bet (0 pts)'}</div>"
                        f"<div style='font-size: 0.75rem; color: #94a3b8;'>"
                        f"Probability: {p_close:.2%} | Kelly Fraction: {c_f:.1%}"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    if c_cut_stake > 0:
                        st.markdown(
                            f"<div style='padding: 10px; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 10px; margin-top: 10px;'>"
                            f"<span style='font-size: 0.75rem; color: #cbd5e1;'>Hedge Close Cut ({close_cut_digit}): <b>{c_cut_stake:.1f} pts</b> (Prob: {p_close_cut:.1%})</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div style='padding: 10px; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 10px; margin-top: 10px;'>"
                            f"<span style='font-size: 0.75rem; color: #64748b;'>No hedge recommended for Close Cut ({close_cut_digit})</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
            else:
                st.markdown("<h5 style='color: #cbd5e1; font-family: Space Grotesk, sans-serif; font-size: 1rem; margin-top: 15px;'>🎯 Recommended Stakes: Jodi / High Odds</h5>", unsafe_allow_html=True)
                col_j1, col_j2 = st.columns(2)
                
                with col_j1:
                    st.markdown(
                        f"<div style='padding: 15px; background: rgba(255, 140, 0, 0.05); border: 1px solid rgba(255, 140, 0, 0.15); border-radius: 12px; height: 100%;'>"
                        f"<div style='font-size: 0.8rem; color: #ffa53b; text-transform: uppercase; letter-spacing: 1px;'>Jodi Bet: {pred['predicted_jodi']}</div>"
                        f"<div style='font-size: 1.8rem; font-weight: 700; color: #fff; margin: 5px 0;'>"
                        f"{f'{j_stake:.1f} pts' if j_stake > 0 else 'No Bet (0 pts)'}</div>"
                        f"<div style='font-size: 0.75rem; color: #94a3b8;'>"
                        f"Probability: {p_jodi:.2%} | Kelly Fraction: {j_f:.2%}"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with col_j2:
                    st.markdown(
                        f"<div style='padding: 15px; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; height: 100%;'>"
                        f"<div style='font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;'>Cut Jodi Hedge: {jodi_cut_str}</div>"
                        f"<div style='font-size: 1.8rem; font-weight: 700; color: #fff; margin: 5px 0;'>"
                        f"{f'{j_cut_stake:.1f} pts' if j_cut_stake > 0 else 'No Bet (0 pts)'}</div>"
                        f"<div style='font-size: 0.75rem; color: #94a3b8;'>"
                        f"Probability: {p_jodi_cut:.2%} | Kelly Fraction: {j_cut_f:.2%}"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            st.markdown(
                "<div style='margin-top: 15px; padding: 10px; background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.15); border-radius: 8px;'>"
                "<span style='font-size: 0.75rem; color: #f87171;'>⚠️ <b>Educational Risk Notice:</b> Kelly Criterion assumes precise probability estimation. Satta Matka is highly volatile. Never wager actual money; use this simulation tool for pattern research only.</span>"
                "</div>",
                unsafe_allow_html=True
            )
        
    with col_prob:
        with st.container(border=True):
            st.markdown('<div class="card-title">📊 Probability Distribution (Digits 0-9)</div>', unsafe_allow_html=True)
            
            digits = [str(x) for x in range(10)]
            prob_df = pd.DataFrame({
                "Digit": digits * 2,
                "Probability": pred["open_probs"] + pred["close_probs"],
                "Draw Phase": ["Open Single"] * 10 + ["Close Single"] * 10
            })
            
            open_chart_df = pd.DataFrame({
                "Digit": [str(d) for d in range(10)],
                "Probability": [round(p * 100, 2) for p in pred["open_probs"]]
            })
            
            open_chart = alt.Chart(open_chart_df).mark_bar(
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4,
                color="#ff8c00"
            ).encode(
                x=alt.X('Digit:O', axis=alt.Axis(title='Digit')),
                y=alt.Y('Probability:Q', axis=alt.Axis(title='Probability (%)')),
                tooltip=['Digit', 'Probability']
            ).properties(
                title="Open Single Digit Probabilities",
                height=200
            )
            
            st.altair_chart(open_chart, use_container_width=True)
            
            close_chart_df = pd.DataFrame({
                "Digit": [str(d) for d in range(10)],
                "Probability": [round(p * 100, 2) for p in pred["close_probs"]]
            })
            
            close_chart = alt.Chart(close_chart_df).mark_bar(
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4,
                color="#ff0055"
            ).encode(
                x=alt.X('Digit:O', axis=alt.Axis(title='Digit')),
                y=alt.Y('Probability:Q', axis=alt.Axis(title='Probability (%)')),
                tooltip=['Digit', 'Probability']
            ).properties(
                title="Close Single Digit Probabilities",
                height=200
            )
            
            st.altair_chart(close_chart, use_container_width=True)
            
        # Get baseline backtest results
        try:
            baseline_res = engine.backtest(test_draws_count=30)
            with st.container(border=True):
                render_html(f"""
                    <div class="card-title">📈 Model Accuracy (30-Draw Backtest)</div>
                    <p style="color: #94a3b8; font-size: 0.85rem; margin-top: -10px; margin-bottom: 20px;">
                        Historical prediction accuracy measured on the last 30 draws chronologically (no lookahead bias).
                    </p>
                    <div class="stats-grid">
                        <div class="stat-mini-box" style="border-color: rgba(34, 197, 94, 0.15);">
                            <div class="stat-mini-val" style="color: #22c55e;">{baseline_res['accuracy_open']:.2%}</div>
                            <div class="stat-mini-lbl">Open Accuracy</div>
                        </div>
                        <div class="stat-mini-box" style="border-color: rgba(34, 197, 94, 0.15);">
                            <div class="stat-mini-val" style="color: #22c55e;">{baseline_res['accuracy_close']:.2%}</div>
                            <div class="stat-mini-lbl">Close Accuracy</div>
                        </div>
                        <div class="stat-mini-box" style="border-color: rgba(59, 130, 246, 0.15);">
                            <div class="stat-mini-val" style="color: #3b82f6;">{baseline_res['top3_rate_open']:.2%}</div>
                            <div class="stat-mini-lbl">Open Top-3 Hit</div>
                        </div>
                        <div class="stat-mini-box" style="border-color: rgba(59, 130, 246, 0.15);">
                            <div class="stat-mini-val" style="color: #3b82f6;">{baseline_res['top3_rate_close']:.2%}</div>
                            <div class="stat-mini-lbl">Close Top-3 Hit</div>
                        </div>
                    </div>
                    <div style="margin-top: 15px; text-align: center;">
                        <span style="font-size: 0.8rem; color: #64748b;">
                            Exact Jodi Accuracy: <b>{baseline_res['accuracy_jodi']:.2%}</b> (Random Baseline: 1.00%)
                        </span>
                    </div>
                """)
        except Exception as e:
            st.warning(f"Could not load baseline backtest metrics: {e}")

    # Reverse Solver Card
    if is_past_date and actual_row is not None:
        matches = find_matching_formulas(df_valid, target_date)
        if matches:
            with st.container(border=True):
                st.markdown('<div class="card-title">🧮 Reverse-Engineered Formula Solver</div>', unsafe_allow_html=True)
                st.markdown("<p style='color: #94a3b8; font-size: 0.85rem; margin-top: -10px;'>These mathematical relationships evaluate exactly to target draw's digits from previous result variables.</p>", unsafe_allow_html=True)
                
                c_m1, c_m2 = st.columns(2)
                with c_m1:
                    st.markdown(f"<span style='color: #ff8c00; font-weight: 600; font-size: 0.95rem;'>Open Single Match ({actual_row['open_single']})</span>", unsafe_allow_html=True)
                    if matches["openMatches"]:
                        for m in matches["openMatches"]:
                            st.markdown(f"formula: `{m['expression']}` = **{m['result']}**")
                    else:
                        st.markdown("<span style='color:#64748b;'>No simple formula matched today's open.</span>", unsafe_allow_html=True)
                with c_m2:
                    st.markdown(f"<span style='color: #ff0055; font-weight: 600; font-size: 0.95rem;'>Close Single Match ({actual_row['close_single']})</span>", unsafe_allow_html=True)
                    if matches["closeMatches"]:
                        for m in matches["closeMatches"]:
                            st.markdown(f"formula: `{m['expression']}` = **{m['result']}**")
                    else:
                        st.markdown("<span style='color:#64748b;'>No simple formula matched today's close.</span>", unsafe_allow_html=True)

    # Leaderboard card
    with st.container(border=True):
        if is_past_date:
            st.markdown('<div class="card-title">🏆 All-Model Performance Leaderboard</div>', unsafe_allow_html=True)
            st.markdown("<p style='color: #94a3b8; font-size: 0.85rem; margin-top: -10px;'>Hit/Miss breakdown of all prediction schemes on the selected date.</p>", unsafe_allow_html=True)
            
            results = evaluate_all_models_for_date(df_valid, target_date)
            if results:
                lead_df = pd.DataFrame(results)
                lead_df_display = lead_df[["name", "touchHit", "singleHit", "jodiHit", "panaHit"]].copy()
                lead_df_display.columns = ["Model Name", "Touch Hit", "Single Hit", "Jodi Hit", "Pana Hit"]
                styled_lead = lead_df_display.style.applymap(
                    highlight_hits, subset=["Touch Hit", "Single Hit", "Jodi Hit", "Pana Hit"]
                )
                st.dataframe(styled_lead, use_container_width=True, hide_index=True)
        else:
            st.markdown('<div class="card-title">🏆 Model Leaderboard: Future Predictions</div>', unsafe_allow_html=True)
            st.markdown("<p style='color: #94a3b8; font-size: 0.85rem; margin-top: -10px;'>Compare prediction digits and suggestions across all models for tomorrow's draw.</p>", unsafe_allow_html=True)
            
            results = get_all_model_predictions_for_future(df_valid, target_date)
            if results:
                lead_df = pd.DataFrame(results)
                lead_df["Touch Digits"] = lead_df["touchDigits"].apply(lambda x: ", ".join(x))
                lead_df["Predicted Jodi"] = lead_df["jodis"].apply(lambda x: x[0] if x else "N/A")
                lead_df["Suggested Panas"] = lead_df["panas"].apply(lambda x: ", ".join(x[:3]))
                lead_df_display = lead_df[["name", "Touch Digits", "Predicted Jodi", "Suggested Panas"]].copy()
                lead_df_display.columns = ["Model Name", "Touch Digits", "Predicted Jodi", "Suggested Panas"]
                st.dataframe(lead_df_display, use_container_width=True, hide_index=True)

    # Recent results ribbon
    with st.container(border=True):
        st.markdown('<div class="card-title">🎴 Recent Draw Results (Last 10 Days)</div>', unsafe_allow_html=True)
        
        recent_draws = df_valid.tail(10).iloc[::-1]
        cols = st.columns(10)
        for idx, (_, row) in enumerate(recent_draws.iterrows()):
            with cols[idx]:
                render_html(f"""
                    <div style='text-align:center; padding: 12px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); border-radius: 12px;'>
                        <div style='font-size:0.75rem; color:#94a3b8; font-weight:500;'>{row['date'].strftime('%d %b')}</div>
                        <div style='font-size:0.7rem; color:#64748b; margin-bottom: 6px;'>{row['weekday'][:3]}</div>
                        <div style='font-size:1.4rem; font-weight:800; color:#ff8c00; font-family:Space Grotesk, sans-serif; letter-spacing:1px;'>{row['jodi']}</div>
                        <div style='font-size:0.7rem; color:#cbd5e1; font-family:monospace; margin-top:4px;'>{row['open_pana']}-{row['close_pana']}</div>
                    </div>
                """)
                
    # AI Analyst Report Container
    with st.container(border=True):
        st.markdown('<div class="card-title">🔮 LLM-Powered AI Analyst Insights</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color: #94a3b8; font-size: 0.85rem; margin-top: -10px; margin-bottom: 15px;">'
            'Generate a real-time narrative report summarizing predictions, model weights, and hedging advice using generative AI.'
            '</p>',
            unsafe_allow_html=True
        )
        
        if not api_key_input:
            st.info("🔑 Please enter your Google Gemini API Key in the sidebar's **AI Analyst Setup** section to enable AI Insights.")
        else:
            if "ai_report_cache" not in st.session_state:
                st.session_state.ai_report_cache = ""
                
            col_btn, col_info = st.columns([1, 4])
            with col_btn:
                trigger_btn = st.button("🔮 Generate AI Report", type="primary", use_container_width=True)
            with col_info:
                st.markdown(
                    f"<div style='color: #a855f7; font-size: 0.85rem; margin-top: 6px; font-weight: 500;'>"
                    f"Selected Model: <b>{ai_model}</b>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
            if trigger_btn:
                with st.spinner("Analyzing predictions, patterns, and risk matrices..."):
                    hist_summary_str = get_history_summary(df_valid)
                    if 'kelly_stakes_dict' not in locals():
                        kelly_stakes_dict = {}
                    report = generate_ai_report(
                        prediction=pred,
                        history_summary=hist_summary_str,
                        model_weights=weights,
                        kelly_stakes=kelly_stakes_dict,
                        api_key=api_key_input,
                        model_name=ai_model
                    )
                    st.session_state.ai_report_cache = report
                    
            if st.session_state.ai_report_cache:
                st.markdown("---")
                st.markdown(
                    f"<div style='background: rgba(255, 255, 255, 0.01); border: 1px solid rgba(255, 255, 255, 0.03); border-radius: 12px; padding: 20px; font-size: 0.95rem; line-height: 1.6;'>",
                    unsafe_allow_html=True
                )
                st.markdown(st.session_state.ai_report_cache)
                st.markdown("</div>", unsafe_allow_html=True)

# ==================== VIEW 2: CHARTS & ANALYTICS ====================
elif nav_selection == "📊 Charts & Analytics":
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        with st.container(border=True):
            st.markdown('<div class="card-title">📊 Digits Frequency Distribution</div>', unsafe_allow_html=True)
            
            # Build frequency dataframe
            freq_df = pd.DataFrame({
                "Digit": [str(d) for d in range(10)],
                "Open Count": [hist_stats["open_freq"].get(d, 0) for d in range(10)],
                "Close Count": [hist_stats["close_freq"].get(d, 0) for d in range(10)]
            })
            
            freq_melted = freq_df.melt(id_vars="Digit", var_name="Type", value_name="Count")
            
            freq_chart = alt.Chart(freq_melted).mark_bar(
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3
            ).encode(
                x=alt.X('Digit:O', axis=alt.Axis(title='Digits 0-9')),
                y=alt.Y('Count:Q', axis=alt.Axis(title='Frequency Count')),
                color=alt.Color('Type:N', scale=alt.Scale(domain=['Open Count', 'Close Count'], range=['#ff8c00', '#ff0055']), legend=alt.Legend(title="Draw Phase")),
                xOffset='Type:N',
                tooltip=['Digit', 'Type', 'Count']
            ).properties(
                height=250
            )
            
            st.altair_chart(freq_chart, use_container_width=True)
        
        # Stats summary boxes
        with st.container(border=True):
            st.markdown('<div class="card-title">🔥 Digit Heat Indicators</div>', unsafe_allow_html=True)
            
            h_open_str = ", ".join(map(str, hist_stats["hot_digits"]))
            c_open_str = ", ".join(map(str, hist_stats["cold_digits"]))
            
            render_html(f"""
                <div class="stats-grid">
                    <div class="stat-mini-box">
                        <div class="stat-mini-val" style="color: #22c55e;">{h_open_str}</div>
                        <div class="stat-mini-lbl">Hot Digits</div>
                    </div>
                    <div class="stat-mini-box">
                        <div class="stat-mini-val" style="color: #ef4444;">{c_open_str}</div>
                        <div class="stat-mini-lbl">Cold Digits</div>
                    </div>
                    <div class="stat-mini-box">
                        <div class="stat-mini-val" style="color: #ff8c00;">{hist_stats['cut_numbers_cooccurrence_rate']:.1%}</div>
                        <div class="stat-mini-lbl">Cut Jodi Rate</div>
                    </div>
                    <div class="stat-mini-box">
                        <div class="stat-mini-val">{hist_stats['autocorrelation']['open']:.3f}</div>
                        <div class="stat-mini-lbl">Open Autocorr</div>
                    </div>
                </div>
            """)
        
    with col_r:
        with st.container(border=True):
            render_html("""
                <div class="card-title">📊 Weekday-Specific Heatmap</div>
                <p style="color: #94a3b8; font-size: 0.85rem; margin-top: -10px;">
                    Frequencies of Open Single digits across weekdays to isolate weekly pattern bias.
                </p>
            """)
            
            # Weekday heatmap table in raw HTML for premium styling
            weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            # Build 10x7 array
            heatmap_data = np.zeros((10, 7))
            for day_idx, day_name in enumerate(weekdays):
                day_freqs = hist_stats["weekday_open_freq"].get(day_name, {})
                for d in range(10):
                    heatmap_data[d][day_idx] = day_freqs.get(d, 0)
                    
            max_val = heatmap_data.max() if heatmap_data.max() > 0 else 1
            
            # Render HTML
            html = "<table class='heatmap-table'><thead><tr><th>Digit</th>"
            for w in weekdays:
                html += f"<th>{w[:3]}</th>"
            html += "</tr></thead><tbody>"
            
            for d in range(10):
                html += f"<tr><td class='heatmap-row-label'>{d}</td>"
                for w_idx in range(7):
                    count = int(heatmap_data[d][w_idx])
                    opacity = count / max_val
                    # Dark mode appropriate orange styling
                    bg_color = f"rgba(255, 140, 0, {opacity * 0.75 + 0.05})"
                    text_color = "#fff" if opacity > 0.4 else "#94a3b8"
                    html += f"<td style='background: {bg_color}; color: {text_color};' title='Digit {d} occurred {count} times on {weekdays[w_idx]}'>{count}</td>"
                html += "</tr>"
                
            html += "</tbody></table>"
            st.markdown(html, unsafe_allow_html=True)
        
    # Panna types breakdown
    with st.container(border=True):
        st.markdown('<div class="card-title">📦 Panna Classification Breakdown (Patti Patterns)</div>', unsafe_allow_html=True)
        
        col_p1, col_p2 = st.columns(2)
        
        # Open Panna Classification
        p_types_o = hist_stats["panna_types"]["open"]
        panna_o_df = pd.DataFrame({
            "Type": [k.capitalize() + " Panna" for k in p_types_o.keys() if k != "unknown"],
            "Count": [v for k, v in p_types_o.items() if k != "unknown"]
        })
        
        with col_p1:
            st.markdown("<p style='text-align:center; font-weight:600; color:#ff8c00;'>Open Pana Types</p>", unsafe_allow_html=True)
            chart_o_p = alt.Chart(panna_o_df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Type", type="nominal", scale=alt.Scale(range=['#3b82f6', '#ff8c00', '#ec4899'])),
                tooltip=["Type", "Count"]
            ).properties(height=200)
            st.altair_chart(chart_o_p, use_container_width=True)
            
        # Close Panna Classification
        p_types_c = hist_stats["panna_types"]["close"]
        panna_c_df = pd.DataFrame({
            "Type": [k.capitalize() + " Panna" for k in p_types_c.keys() if k != "unknown"],
            "Count": [v for k, v in p_types_c.items() if k != "unknown"]
        })
        
        with col_p2:
            st.markdown("<p style='text-align:center; font-weight:600; color:#ff0055;'>Close Pana Types</p>", unsafe_allow_html=True)
            chart_c_p = alt.Chart(panna_c_df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Type", type="nominal", scale=alt.Scale(range=['#3b82f6', '#ff8c00', '#ec4899'])),
                tooltip=["Type", "Count"]
            ).properties(height=200)
            st.altair_chart(chart_c_p, use_container_width=True)

# ==================== VIEW 3: MODEL TUNING & BACKTEST ====================
elif nav_selection == "⚙️ Model Tuning & Backtest":
    tab1, tab2, tab3 = st.tabs([
        "🤖 ML Ensemble Backtest", 
        "🧮 Algebraic Formulas Backtest", 
        "⚡ Formula Auto-Optimizer"
    ])
    
    with tab1:
        with st.container(border=True):
            st.markdown('<div class="card-title">🤖 ML Ensemble Chronological Backtest</div>', unsafe_allow_html=True)
            st.markdown("""
                <p style="color: #94a3b8; font-size: 0.9rem; margin-top: -10px;">
                    Simulates predictions historically over the last <b>N</b> draws using the Machine Learning Ensemble.
                    For each draw, the ensemble model is trained <i>only</i> on prior historical data, ensuring no lookahead bias.
                </p>
            """, unsafe_allow_html=True)
            
            col_back_config, col_back_run = st.columns([1, 2])
            
            with col_back_config:
                st.markdown("<h5 style='color: #cbd5e1; font-family: Space Grotesk, sans-serif; font-size: 1rem;'>Backtest Settings</h5>", unsafe_allow_html=True)
                backtest_n = st.slider("Number of historical draws to test", 10, 100, 30, 5, key="ml_backtest_n")
                
                st.markdown("<br>", unsafe_allow_html=True)
                trigger_backtest = st.button("🚀 Run ML Backtest Simulation", use_container_width=True)
                
            with col_back_run:
                if trigger_backtest:
                    with st.spinner("Running historical ML backtest simulations..."):
                        backtest_res = engine.backtest(test_draws_count=backtest_n, weights=weights)
                        
                        if "error" in backtest_res:
                            st.error(backtest_res["error"])
                        else:
                            st.toast("ML Backtest completed!", icon="✅")
                            
                            render_html(f"""
                                <div class="stats-grid" style="margin-bottom: 25px;">
                                    <div class="stat-mini-box" style="border-color: rgba(34, 197, 94, 0.2);">
                                        <div class="stat-mini-val" style="color: #22c55e;">{backtest_res['accuracy_open']:.2%}</div>
                                        <div class="stat-mini-lbl">Open Accuracy</div>
                                    </div>
                                    <div class="stat-mini-box" style="border-color: rgba(34, 197, 94, 0.2);">
                                        <div class="stat-mini-val" style="color: #22c55e;">{backtest_res['accuracy_close']:.2%}</div>
                                        <div class="stat-mini-lbl">Close Accuracy</div>
                                    </div>
                                    <div class="stat-mini-box" style="border-color: rgba(59, 130, 246, 0.2);">
                                        <div class="stat-mini-val" style="color: #3b82f6;">{backtest_res['top3_rate_open']:.2%}</div>
                                        <div class="stat-mini-lbl">Open Top-3 Hit</div>
                                    </div>
                                    <div class="stat-mini-box" style="border-color: rgba(59, 130, 246, 0.2);">
                                        <div class="stat-mini-val" style="color: #3b82f6;">{backtest_res['top3_rate_close']:.2%}</div>
                                        <div class="stat-mini-lbl">Close Top-3 Hit</div>
                                    </div>
                                    <div class="stat-mini-box" style="border-color: rgba(239, 68, 68, 0.2);">
                                        <div class="stat-mini-val" style="color: #ef4444;">{backtest_res['accuracy_jodi']:.2%}</div>
                                        <div class="stat-mini-lbl">Exact Jodi Hit</div>
                                    </div>
                                </div>
                            """)
                            
                            # Chart cumulative correct
                            detailed_df = pd.DataFrame(backtest_res["detailed_results"])
                            
                            # Convert to cumulative counts
                            detailed_df["cum_hits_open"] = detailed_df["is_hit_open"].cumsum()
                            detailed_df["cum_hits_close"] = detailed_df["is_hit_close"].cumsum()
                            detailed_df["cum_hits_jodi"] = detailed_df["is_hit_jodi"].cumsum()
                            detailed_df["trials"] = range(1, len(detailed_df) + 1)
                            
                            # Melt for Altair line chart
                            cum_df = detailed_df[["trials", "cum_hits_open", "cum_hits_close", "cum_hits_jodi"]].melt(
                                id_vars="trials", var_name="Metric", value_name="Cumulative Correct Hits"
                            )
                            cum_df["Metric"] = cum_df["Metric"].map({
                                "cum_hits_open": "Open Single Hit",
                                "cum_hits_close": "Close Single Hit",
                                "cum_hits_jodi": "Jodi Hit"
                            })
                            
                            cum_chart = alt.Chart(cum_df).mark_line(point=True).encode(
                                x=alt.X("trials:Q", axis=alt.Axis(title="Draw Index")),
                                y=alt.Y("Cumulative Correct Hits:Q", axis=alt.Axis(title="Cumulative Correct Predictions")),
                                color=alt.Color("Metric:N", scale=alt.Scale(range=["#ff8c00", "#ff0055", "#3b82f6"])),
                                tooltip=["trials", "Cumulative Correct Hits", "Metric"]
                            ).properties(height=250)
                            
                            st.altair_chart(cum_chart, use_container_width=True)
                            
                            # Detailed Trials Expander
                            with st.expander("🔍 View Detailed Trial Logs"):
                                log_display_df = detailed_df[[
                                    "date", "weekday", "actual_open", "actual_close", "actual_jodi",
                                    "predicted_open", "predicted_close", "predicted_jodi", 
                                    "is_hit_open", "is_hit_close", "is_hit_jodi"
                                ]].copy()
                                
                                log_display_df.columns = [
                                    "Date", "Weekday", "Act Open", "Act Close", "Act Jodi",
                                    "Pred Open", "Pred Close", "Pred Jodi", "Hit Open", "Hit Close", "Hit Jodi"
                                ]
                                
                                styled_logs = log_display_df.style.applymap(
                                    highlight_hits, subset=["Hit Open", "Hit Close", "Hit Jodi"]
                                )
                                st.dataframe(styled_logs, use_container_width=True)
                else:
                    st.info("💡 Adjust settings and click 'Run ML Backtest Simulation' to check the predictive model accuracy.")

        # Model feature importances (inside tab 1)
        with st.container(border=True):
            st.markdown('<div class="card-title">⚙️ ML Feature Importances</div>', unsafe_allow_html=True)
            try:
                X_o, y_o, X_c, y_c = engine._prepare_ml_features(df_valid)
                feature_names = [
                    "Weekday", "Weekday Sin", "Weekday Cos",
                    "Lag 1 (t-1)", "Lag 2 (t-2)", "Lag 3 (t-3)", "Lag 4 (t-4)", "Lag 5 (t-5)",
                    "Rolling Mean 3", "Rolling Mean 5", "Rolling Std 5"
                ]
                
                tab_rf, tab_gbm = st.tabs(["🌳 Random Forest", "⚡ Gradient Boosting"])
                
                with tab_rf:
                    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                    clf_rf.fit(X_o, y_o)
                    imp_rf_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": clf_rf.feature_importances_
                    }).sort_values("Importance", ascending=False)
                    
                    imp_rf_chart = alt.Chart(imp_rf_df).mark_bar(color="#ff8c00").encode(
                        x=alt.X("Importance:Q", axis=alt.Axis(title="Relative Gini Importance")),
                        y=alt.Y("Feature:N", sort="-x", axis=alt.Axis(title=None)),
                        tooltip=["Feature", "Importance"]
                    ).properties(height=220)
                    st.altair_chart(imp_rf_chart, use_container_width=True)
                    
                with tab_gbm:
                    clf_gbm = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
                    clf_gbm.fit(X_o, y_o)
                    imp_gbm_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": clf_gbm.feature_importances_
                    }).sort_values("Importance", ascending=False)
                    
                    imp_gbm_chart = alt.Chart(imp_gbm_df).mark_bar(color="#ff0055").encode(
                        x=alt.X("Importance:Q", axis=alt.Axis(title="Relative Gini Importance")),
                        y=alt.Y("Feature:N", sort="-x", axis=alt.Axis(title=None)),
                        tooltip=["Feature", "Importance"]
                    ).properties(height=220)
                    st.altair_chart(imp_gbm_chart, use_container_width=True)
            except Exception as e:
                st.write(f"Could not load feature importances: {e}")

    with tab2:
        with st.container(border=True):
            st.markdown('<div class="card-title">🧮 Algebraic Formulas Backtest Simulator</div>', unsafe_allow_html=True)
            st.markdown("""
                <p style="color: #94a3b8; font-size: 0.9rem; margin-top: -10px;">
                    Historically evaluates algebraic Satta Matka formulas to check success rates (Touch, Single, Jodi, and Pana rates).
                </p>
            """, unsafe_allow_html=True)
            
            col_formula_cfg, col_formula_run = st.columns([1, 2])
            
            with col_formula_cfg:
                st.markdown("<h5 style='color: #cbd5e1; font-family: Space Grotesk, sans-serif; font-size: 1rem;'>Formula Settings</h5>", unsafe_allow_html=True)
                formula_options = {
                    "Date-wise Touch Scheme": "date_touch",
                    "Yesterday's Open-Close Sum": "yesterday_sum",
                    "Weekly Repeat Pattern": "weekly_repeat",
                    "Panel Digit Multiplier": "panel_multiplier",
                    "AI Neural Pattern Recognizer": "ai_pattern",
                    "Adaptive Running Line Optimizer": "adaptive_line",
                    "Custom User Formula": "custom"
                }
                selected_backtest_formula = st.selectbox("Formula Type to Backtest", list(formula_options.keys()), index=0)
                formula_type_key = formula_options[selected_backtest_formula]
                
                backtest_custom_formula_str = ""
                if formula_type_key == "custom":
                    backtest_custom_formula_str = st.text_input(
                        "Backtest Custom Formula String",
                        value="(Open + Close) * 2",
                        help="Variables: Open, Close, Jodi, Date, Month. E.g. (Open + Close) * 2"
                    )
                    
                formula_backtest_n = st.slider("Number of historical draws to test", 10, 100, 30, 5, key="formula_backtest_n")
                st.markdown("<br>", unsafe_allow_html=True)
                trigger_formula_backtest = st.button("🚀 Run Formula Backtest", use_container_width=True)
                
            with col_formula_run:
                if trigger_formula_backtest:
                    with st.spinner(f"Running formula backtest for {selected_backtest_formula}..."):
                        formula_res = run_formula_backtest(
                            df_valid, 
                            formula_type_key, 
                            custom_formula_str=backtest_custom_formula_str, 
                            limit=formula_backtest_n
                        )
                        
                        if not formula_res or formula_res["totalTested"] == 0:
                            st.warning("Insufficient data or no wagers could be evaluated.")
                        else:
                            st.toast("Formula backtest completed!", icon="✅")
                            
                            render_html(f"""
                                <div class="stats-grid" style="margin-bottom: 25px;">
                                    <div class="stat-mini-box" style="border-color: rgba(34, 197, 94, 0.2);">
                                        <div class="stat-mini-val" style="color: #22c55e;">{formula_res['touchRate']}%</div>
                                        <div class="stat-mini-lbl">Touch Rate</div>
                                    </div>
                                    <div class="stat-mini-box" style="border-color: rgba(34, 197, 94, 0.2);">
                                        <div class="stat-mini-val" style="color: #22c55e;">{formula_res['singleRate']}%</div>
                                        <div class="stat-mini-lbl">Single Rate</div>
                                    </div>
                                    <div class="stat-mini-box" style="border-color: rgba(59, 130, 246, 0.2);">
                                        <div class="stat-mini-val" style="color: #3b82f6;">{formula_res['jodiRate']}%</div>
                                        <div class="stat-mini-lbl">Jodi Rate</div>
                                    </div>
                                    <div class="stat-mini-box" style="border-color: rgba(239, 68, 68, 0.2);">
                                        <div class="stat-mini-val" style="color: #ef4444;">{formula_res['panaRate']}%</div>
                                        <div class="stat-mini-lbl">Pana Rate</div>
                                    </div>
                                    <div class="stat-mini-box" style="border-color: rgba(255, 255, 255, 0.1);">
                                        <div class="stat-mini-val" style="color: #cbd5e1;">{formula_res['totalTested']}</div>
                                        <div class="stat-mini-lbl">Total Tested</div>
                                    </div>
                                </div>
                            """)
                            
                            # Altair cumulative hits chart
                            detailed_df = pd.DataFrame(formula_res["detailed_results"])
                            detailed_df["cum_hits_touch"] = detailed_df["is_hit_touch"].cumsum()
                            detailed_df["cum_hits_single"] = detailed_df["is_hit_single"].cumsum()
                            detailed_df["cum_hits_jodi"] = detailed_df["is_hit_jodi"].cumsum()
                            detailed_df["cum_hits_pana"] = detailed_df["is_hit_pana"].cumsum()
                            detailed_df["trials"] = range(1, len(detailed_df) + 1)
                            
                            cum_df = detailed_df[["trials", "cum_hits_touch", "cum_hits_single", "cum_hits_jodi", "cum_hits_pana"]].melt(
                                id_vars="trials", var_name="Metric", value_name="Cumulative Correct Hits"
                            )
                            cum_df["Metric"] = cum_df["Metric"].map({
                                "cum_hits_touch": "Touch Hit",
                                "cum_hits_single": "Single Hit",
                                "cum_hits_jodi": "Jodi Hit",
                                "cum_hits_pana": "Pana Hit"
                            })
                            
                            cum_chart = alt.Chart(cum_df).mark_line(point=True).encode(
                                x=alt.X("trials:Q", axis=alt.Axis(title="Draw Index")),
                                y=alt.Y("Cumulative Correct Hits:Q", axis=alt.Axis(title="Cumulative Correct Hits")),
                                color=alt.Color("Metric:N", scale=alt.Scale(range=["#3b82f6", "#22c55e", "#ff8c00", "#ff0055"])),
                                tooltip=["trials", "Cumulative Correct Hits", "Metric"]
                            ).properties(height=250)
                            
                            st.altair_chart(cum_chart, use_container_width=True)
                            
                            # Detailed trial logs expander
                            with st.expander("🔍 View Detailed Formula Logs"):
                                log_display_df = detailed_df[[
                                    "date", "weekday", "actual_open", "actual_close", "actual_jodi",
                                    "actual_open_pana", "actual_close_pana", "predicted_touch", "is_hit_touch", "is_hit_single", "is_hit_jodi", "is_hit_pana"
                                ]].copy()
                                
                                # Format lists for pretty display
                                log_display_df["predicted_touch"] = log_display_df["predicted_touch"].apply(lambda x: ", ".join(x))
                                
                                log_display_df.columns = [
                                    "Date", "Weekday", "Act Open", "Act Close", "Act Jodi",
                                    "Act Open Pana", "Act Close Pana", "Pred Touch", "Hit Touch", "Hit Single", "Hit Jodi", "Hit Pana"
                                ]
                                
                                styled_logs = log_display_df.style.applymap(
                                    highlight_hits, subset=["Hit Touch", "Hit Single", "Hit Jodi", "Hit Pana"]
                                )
                                st.dataframe(styled_logs, use_container_width=True)
                else:
                    st.info("💡 Select a formula type, adjust the settings, and click 'Run Formula Backtest' to review chronological performance.")

    with tab3:
        with st.container(border=True):
            st.markdown('<div class="card-title">⚡ Formula Auto-Optimizer</div>', unsafe_allow_html=True)
            st.markdown("""
                <p style="color: #94a3b8; font-size: 0.9rem; margin-top: -10px;">
                    Scans 60+ algebraic templates historically over the recent draws and ranks them dynamically by **Touch Rate** to discover the most optimized formula for the current market trend.
                </p>
            """, unsafe_allow_html=True)
            
            col_opt_cfg, col_opt_run = st.columns([1, 2])
            
            with col_opt_cfg:
                st.markdown("<h5 style='color: #cbd5e1; font-family: Space Grotesk, sans-serif; font-size: 1rem;'>Optimization Scope</h5>", unsafe_allow_html=True)
                opt_depth = st.selectbox(
                    "Backtest Depth (Past Draws)", 
                    [15, 30, 50], 
                    index=1,
                    help="Number of historical draws to evaluate each formula template against."
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                trigger_opt = st.button("⚡ Scan & Optimize Formulas", use_container_width=True)
                
            with col_opt_run:
                if trigger_opt or "best_formulas_cache" in st.session_state:
                    if trigger_opt:
                        with st.spinner("Scrambling, scanning and backtesting 60+ arithmetic templates..."):
                            best_formulas = find_best_custom_formula(df_valid, depth=opt_depth)
                            st.session_state.best_formulas_cache = best_formulas
                            st.session_state.opt_depth_used = opt_depth
                            st.toast("Formula scanning complete!", icon="🔥")
                    
                    best_formulas = st.session_state.best_formulas_cache
                    depth_used = st.session_state.opt_depth_used
                    
                    st.markdown(f"<h5 style='color: #cbd5e1; font-family: Space Grotesk, sans-serif; font-size: 1.1rem; margin-bottom: 15px;'>🏆 Top 10 Ranked Algebraic Formulas (Depth: {depth_used} draws)</h5>", unsafe_allow_html=True)
                    
                    # Display top 10 formulas in a grid with an apply button
                    for idx, formula in enumerate(best_formulas):
                        with st.container():
                            c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1.5])
                            c1.markdown(f"**#{idx+1}** &nbsp; ` {formula['expression']} `", unsafe_allow_html=True)
                            c2.markdown(f"<span style='color: #22c55e;'>Touch: **{formula['touchRate']}%**</span>", unsafe_allow_html=True)
                            c3.markdown(f"<span style='color: #3b82f6;'>Jodi: **{formula['jodiRate']}%**</span>", unsafe_allow_html=True)
                            c4.markdown(f"<span style='color: #ec4899;'>Pana: **{formula['panaRate']}%**</span>", unsafe_allow_html=True)
                            
                            # Render Apply Button
                            c5.button(
                                "Apply ⚡",
                                key=f"apply_formula_{idx}",
                                on_click=apply_formula_callback,
                                args=(formula['expression'],),
                                use_container_width=True
                            )
                                
                            st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px solid rgba(255,255,255,0.03);'>", unsafe_allow_html=True)
                else:
                    st.info("💡 Click 'Scan & Optimize Formulas' to search Satta Matka numerology templates for the best mathematical match.")

# ==================== VIEW 4: HISTORICAL DATA EXPLORER ====================
elif nav_selection == "🗂️ Historical Data Explorer":
    with st.container(border=True):
        render_html("""
            <div class="card-title">🗂️ Historical Records Explorer</div>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: -10px;">
                Full Satta Matka Mahadevi historical database. Missing draws/market holidays are highlighted.
            </p>
        """)
        
        # Filter controls
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            f_search = st.text_input("Search Jodi / Pana (e.g. '97' or '234')")
        with col_f2:
            f_weekday = st.multiselect("Filter by Weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        with col_f3:
            f_valid_only = st.checkbox("Show valid draws only", value=False)
            
        # Copy of data
        df_filtered = df_raw.copy()
        
        # Apply filters
        if f_search:
            df_filtered = df_filtered[
                (df_filtered["jodi"].str.contains(f_search, na=False)) |
                (df_filtered["open_pana"].str.contains(f_search, na=False)) |
                (df_filtered["close_pana"].str.contains(f_search, na=False))
            ]
            
        if f_weekday:
            df_filtered = df_filtered[df_filtered["weekday"].isin(f_weekday)]
            
        if f_valid_only:
            df_filtered = df_filtered[df_filtered["is_valid"] == True]
            
        # Sort descending for explorer (newest first)
        df_filtered = df_filtered.sort_values("date", ascending=False)
        
        # Formatting display
        display_df = df_filtered.copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        
        display_df = display_df[[
            "date", "weekday", "open_pana", "jodi", "close_pana", "open_single", "close_single", "is_valid"
        ]]
        
        display_df.columns = [
            "Date", "Weekday", "Open Pana", "Jodi", "Close Pana", "Open Single", "Close Single", "Is Valid"
        ]
        
        # Table styling for UI
        def highlight_validity(row):
            is_valid = row["Is Valid"]
            if is_valid is False:
                return ['background-color: rgba(239, 68, 68, 0.08); color: #94a3b8; font-style: italic;'] * len(row)
            return [''] * len(row)
            
        styled_df = display_df.style.apply(highlight_validity, axis=1)
        
        st.dataframe(styled_df, use_container_width=True, height=500)
