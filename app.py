import io
import pickle
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent


@st.cache_resource
def load_artifacts():
    with open(ROOT / "model.pkl", "rb") as file:
        model = pickle.load(file)
    with open(ROOT / "label_encoders.pkl", "rb") as file:
        encoders = pickle.load(file)
    with open(ROOT / "feature_columns.pkl", "rb") as file:
        columns = pickle.load(file)
    return model, encoders, columns


@st.cache_data
def load_reference_data():
    data = pd.read_csv(ROOT / "dataset.csv")
    return data.sample(min(15000, len(data)), random_state=42)


def load_prediction_history():
    hist_path = ROOT / "prediction_history.csv"
    if hist_path.exists() and hist_path.stat().st_size > 0:
        try:
            df = pd.read_csv(hist_path, header=None, names=["Student ID", "Predicted Score"])
            df["Student ID"] = pd.to_numeric(df["Student ID"], errors='coerce')
            df["Predicted Score"] = pd.to_numeric(df["Predicted Score"], errors='coerce')
            df = df.dropna()
            df["Student ID"] = df["Student ID"].astype(int)
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["Student ID", "Predicted Score"])


def append_prediction_history(student_id, score):
    hist_path = ROOT / "prediction_history.csv"
    df = load_prediction_history()
    new_row = pd.DataFrame([{"Student ID": int(student_id), "Predicted Score": float(score)}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(hist_path, index=False, header=False)
    return df


def generate_report_file(values, score, level, file_type, profile_name="Student"):
    student_id = values.get("Student_ID", 1)
    student_name = profile_name or "Student"
    
    if file_type == "CSV":
        report_df = pd.DataFrame([{
            "Student_ID": student_id,
            "Student_Name": student_name,
            "Predicted_Final_Score": score,
            "Performance_Level": level,
            "Attendance_Rate": values.get("Attendance_Rate", 0),
            **values
        }])
        return report_df.to_csv(index=False).encode('utf-8'), f"prediction_report_{student_id}.csv", "text/csv"
    
    elif file_type == "HTML":
        generated_at = datetime.now().strftime("%d %b %Y, %I:%M %p")
        html_content = f"""<!doctype html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Prediction Report - {escape(student_name)}</title>
    <style>
        body {{ font-family: 'Outfit', Arial, sans-serif; background: #070a13; color: #f8fafc; padding: 40px; max-width: 800px; margin: auto; }}
        .card {{ background: rgba(15, 23, 42, 0.9); border: 1px solid #06b6d4; border-radius: 20px; padding: 32px; box-shadow: 0 10px 40px rgba(6,182,212,0.2); }}
        h1 {{ color: #06b6d4; margin-top: 0; }}
        .score {{ font-size: 56px; font-weight: bold; color: #10b981; margin: 10px 0; }}
        .badge {{ display: inline-block; padding: 6px 16px; border-radius: 20px; background: rgba(6,182,212,0.2); color: #38bdf8; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 24px; }}
        td {{ padding: 10px; border-bottom: 1px solid #334155; }}
        .muted {{ color: #94a3b8; font-size: 0.85rem; }}
    </style>
</head>
<body>
    <div class='card'>
        <h1>✦ Student Performance Prediction Report</h1>
        <p>Academic Forecast for <b>{escape(student_name)}</b> (Student ID: {student_id})</p>
        <div class='score'>{score:.2f} / 100</div>
        <div class='badge'>Performance Level: {level}</div>
        <h3>Student Input Overview</h3>
        <table>
            {''.join(f'<tr><td><b>{escape(k.replace("_", " "))}</b></td><td>{escape(str(v))}</td></tr>' for k,v in values.items())}
        </table>
        <p class='muted'>Generated: {generated_at} | Model snapshot for academic planning support.</p>
    </div>
</body>
</html>"""
        return html_content.encode('utf-8'), f"prediction_report_{student_id}.html", "text/html"
    
    elif file_type in ["JPEG", "PDF"]:
        fig, ax = plt.subplots(figsize=(8.5, 7), facecolor='#0f172a')
        ax.set_facecolor('#0f172a')
        ax.axis('off')
        
        ax.text(0.05, 0.92, "✦ Student Performance Prediction Report", color='#06b6d4', fontsize=18, fontweight='bold')
        ax.text(0.05, 0.84, f"Student: {student_name} (ID: {student_id})", color='#94a3b8', fontsize=12)
        
        ax.text(0.05, 0.73, f"Predicted Final Score: {score:.2f} / 100", color='#10b981', fontsize=22, fontweight='bold')
        ax.text(0.05, 0.65, f"Performance Level: {level}", color='#f59e0b', fontsize=14, fontweight='bold')
        ax.text(0.05, 0.58, f"Attendance Rate: {values.get('Attendance_Rate', 0):.1f}%", color='#38bdf8', fontsize=12)
        
        ax.text(0.05, 0.49, "Key Input Signals:", color='#f8fafc', fontsize=13, fontweight='bold')
        y_pos = 0.43
        key_inputs = [
            ("Midterm Mark", values.get("Midterm_Mark", 0)),
            ("Weekly Study Hours", values.get("Weekly_Study_Hours", 0)),
            ("Sleep Hours", values.get("Sleep_Hours", 0)),
            ("Motivation Score", values.get("Motivation_Score", 0)),
            ("Previous GPA", values.get("Previous_GPA", 0)),
            ("Stress Level", values.get("Stress_Level", 0))
        ]
        for k, v in key_inputs:
            ax.text(0.08, y_pos, f"• {k}: {v}", color='#cbd5e1', fontsize=11)
            y_pos -= 0.05
            
        ax.text(0.05, 0.06, "StudyPulse AI Performance Predictor", color='#64748b', fontsize=9, style='italic')
        
        buf = io.BytesIO()
        if file_type == "JPEG":
            fig.savefig(buf, format='jpeg', bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)
            return buf.getvalue(), f"prediction_report_{student_id}.jpg", "image/jpeg"
        else:
            fig.savefig(buf, format='pdf', bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)
            return buf.getvalue(), f"prediction_report_{student_id}.pdf", "application/pdf"


def generate_history_file(history_df, file_type):
    if file_type == "CSV":
        return history_df.to_csv(index=False).encode('utf-8'), "prediction_history.csv", "text/csv"
    
    elif file_type == "HTML":
        rows_html = "".join(f"<tr><td>{row['Student ID']}</td><td>{float(row['Predicted Score']):.4f}</td></tr>" for _, row in history_df.iterrows())
        html_content = f"""<!doctype html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Prediction History</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #070a13; color: #f8fafc; padding: 30px; }}
        table {{ width: 100%; border-collapse: collapse; background: #0f172a; border-radius: 12px; overflow: hidden; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #1e293b; color: #06b6d4; }}
    </style>
</head>
<body>
    <h2>✦ Prediction History</h2>
    <table>
        <thead><tr><th>Student ID</th><th>Predicted Score</th></tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
</body>
</html>"""
        return html_content.encode('utf-8'), "prediction_history.html", "text/html"
    
    elif file_type in ["JPEG", "PDF"]:
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='#0f172a')
        ax.set_facecolor('#0f172a')
        ax.axis('off')
        
        ax.text(0.05, 0.90, "✦ Prediction History", color='#06b6d4', fontsize=18, fontweight='bold')
        
        y_pos = 0.78
        ax.text(0.08, y_pos, "Student ID", color='#94a3b8', fontsize=12, fontweight='bold')
        ax.text(0.55, y_pos, "Predicted Score", color='#94a3b8', fontsize=12, fontweight='bold')
        
        for _, row in history_df.tail(12).iterrows():
            y_pos -= 0.06
            ax.text(0.08, y_pos, str(row['Student ID']), color='#f8fafc', fontsize=11)
            ax.text(0.55, y_pos, f"{float(row['Predicted Score']):.4f}", color='#10b981', fontsize=11)
            
        buf = io.BytesIO()
        if file_type == "JPEG":
            fig.savefig(buf, format='jpeg', bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)
            return buf.getvalue(), "prediction_history.jpg", "image/jpeg"
        else:
            fig.savefig(buf, format='pdf', bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)
            return buf.getvalue(), "prediction_history.pdf", "application/pdf"


model, label_encoders, feature_columns = load_artifacts()
reference = load_reference_data()

# Custom Swasthya Sakha Inspired Moving Background & Modern Glass UI Style
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        :root {
            --bg-dark: #070a13;
            --primary-cyan: #06b6d4;
            --primary-emerald: #10b981;
            --primary-gradient: linear-gradient(135deg, #06b6d4 0%, #10b981 100%);
            --accent-purple: #8b5cf6;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --glass-bg: rgba(15, 23, 42, 0.65);
            --glass-bg-hover: rgba(30, 41, 59, 0.75);
            --glass-border: rgba(255, 255, 255, 0.1);
            --glass-border-glow: rgba(6, 182, 212, 0.4);
            --card-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        /* Root & Page Reset */
        html, body, .stApp {
            background-color: var(--bg-dark) !important;
            color: var(--text-primary);
            font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif;
            overflow-x: hidden;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 1.2rem;
            padding-bottom: 4rem;
            position: relative;
            z-index: 2;
        }

        /* Dynamic Moving Web Background Layer */
        .moving-bg-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            pointer-events: none;
            z-index: 0;
            overflow: hidden;
        }

        .bg-orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.38;
            will-change: transform;
        }

        .bg-orb-1 {
            width: 550px;
            height: 550px;
            top: -100px;
            left: -100px;
            background: radial-gradient(circle, rgba(6, 182, 212, 0.35) 0%, rgba(6, 182, 212, 0) 70%);
            animation: floatOrb1 16s ease-in-out infinite alternate;
        }

        .bg-orb-2 {
            width: 500px;
            height: 500px;
            bottom: -50px;
            right: -50px;
            background: radial-gradient(circle, rgba(16, 185, 129, 0.28) 0%, rgba(16, 185, 129, 0) 70%);
            animation: floatOrb2 20s ease-in-out infinite alternate-reverse;
        }

        .bg-orb-3 {
            width: 450px;
            height: 450px;
            top: 40%;
            right: 25%;
            background: radial-gradient(circle, rgba(139, 92, 246, 0.22) 0%, rgba(139, 92, 246, 0) 70%);
            animation: floatOrb3 18s ease-in-out infinite alternate;
        }

        .bg-orb-4 {
            width: 400px;
            height: 400px;
            bottom: 20%;
            left: 15%;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.25) 0%, rgba(59, 130, 246, 0) 70%);
            animation: floatOrb4 15s ease-in-out infinite alternate-reverse;
        }

        @keyframes floatOrb1 {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(60px, 80px) scale(1.15); }
        }

        @keyframes floatOrb2 {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(-70px, -60px) scale(1.1); }
        }

        @keyframes floatOrb3 {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(-50px, 60px) scale(1.2); }
        }

        @keyframes floatOrb4 {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(50px, -40px) scale(1.05); }
        }

        /* Glassmorphism Header Bar */
        .glass-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.8rem 1.4rem;
            background: rgba(15, 23, 42, 0.7);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            margin-bottom: 1.8rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        }

        .brand-logo {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.3rem;
            font-weight: 800;
            color: #fff;
            letter-spacing: -0.02em;
        }

        .brand-icon-box {
            width: 42px;
            height: 42px;
            border-radius: 14px;
            background: var(--primary-gradient);
            display: grid;
            place-items: center;
            color: #fff;
            font-size: 1.3rem;
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.5);
            animation: pulseGlow 3s ease-in-out infinite;
        }

        .brand-badge {
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.35rem 0.75rem;
            border-radius: 30px;
            background: rgba(6, 182, 212, 0.15);
            border: 1px solid rgba(6, 182, 212, 0.35);
            color: var(--primary-cyan);
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        /* Swasthya Sakha Auth Screen Styling */
        .auth-card-left {
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 28px;
            padding: 3rem 2.8rem;
            position: relative;
            overflow: hidden;
            box-shadow: var(--card-shadow);
            min-height: 540px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .auth-card-left::before {
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 10% 20%, rgba(6, 182, 212, 0.15) 0%, transparent 60%);
            pointer-events: none;
        }

        .splash-logo-box {
            margin: 0 0 1.6rem 0;
            width: 84px;
            height: 84px;
            position: relative;
            animation: logoEntrance 0.9s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
        }

        .splash-logo-glow {
            position: absolute;
            inset: -12px;
            border-radius: 28px;
            background: var(--primary-gradient);
            opacity: 0.5;
            filter: blur(16px);
            animation: pulseGlow 2.5s ease-in-out infinite;
        }

        .splash-logo-card {
            width: 84px;
            height: 84px;
            border-radius: 22px;
            background: linear-gradient(135deg, #0b1329 0%, #1e293b 100%);
            border: 1px solid rgba(6, 182, 212, 0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            z-index: 3;
            animation: logoFloat 4s ease-in-out infinite;
            font-size: 2.2rem;
            color: var(--primary-cyan);
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }

        @keyframes logoEntrance {
            0% { transform: scale(0.3) translateY(30px); opacity: 0; }
            100% { transform: scale(1) translateY(0); opacity: 1; }
        }

        @keyframes logoFloat {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-8px); }
        }

        @keyframes pulseGlow {
            0%, 100% { transform: scale(0.95); opacity: 0.4; }
            50% { transform: scale(1.12); opacity: 0.75; }
        }

        .auth-title {
            font-size: 2.7rem;
            font-weight: 800;
            line-height: 1.12;
            letter-spacing: -0.03em;
            margin: 0.8rem 0 1rem;
            background: linear-gradient(135deg, #ffffff 30%, var(--primary-cyan) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .auth-subtitle {
            font-size: 1.05rem;
            color: var(--text-secondary);
            line-height: 1.65;
            max-width: 440px;
            margin-bottom: 2rem;
        }

        .auth-card-right {
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(24px);
            border: 1px solid rgba(6, 182, 212, 0.25);
            border-radius: 28px;
            padding: 2.5rem 2.2rem;
            box-shadow: 0 25px 60px rgba(0, 0, 0, 0.6), 0 0 40px rgba(6, 182, 212, 0.12);
        }

        .auth-form-header {
            margin-bottom: 1.5rem;
        }

        .auth-form-header h2 {
            font-size: 1.75rem;
            font-weight: 800;
            color: #fff;
            margin-bottom: 0.3rem;
            letter-spacing: -0.02em;
        }

        .auth-form-header p {
            color: var(--text-secondary);
            font-size: 0.92rem;
        }

        /* Hero Section */
        .hero-panel {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.6) 100%);
            backdrop-filter: blur(20px);
            border-radius: 28px;
            padding: 3rem 3.2rem;
            border: 1px solid var(--glass-border);
            margin-bottom: 1.8rem;
            position: relative;
            overflow: hidden;
            box-shadow: var(--card-shadow);
        }

        .hero-panel::after {
            content: '✦';
            position: absolute;
            right: 4%;
            bottom: -30px;
            font-size: 14rem;
            color: var(--primary-cyan);
            opacity: 0.06;
            pointer-events: none;
            transform: rotate(12deg);
        }

        .hero-eyebrow {
            color: var(--primary-cyan);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .hero-title {
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: #fff;
            line-height: 1.15;
            margin-bottom: 0.8rem;
            max-width: 680px;
        }

        .hero-desc {
            color: var(--text-secondary);
            font-size: 1.08rem;
            line-height: 1.65;
            max-width: 580px;
        }

        /* Mini Stat Cards */
        .stat-card {
            background: rgba(15, 23, 42, 0.7);
            backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 1.3rem 1.5rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
            height: 100%;
        }

        .stat-card:hover {
            border-color: rgba(6, 182, 212, 0.4);
            transform: translateY(-3px);
            box-shadow: 0 15px 35px rgba(6, 182, 212, 0.15);
        }

        .stat-label {
            color: var(--primary-cyan);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }

        .stat-value {
            color: #fff;
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }

        .stat-sub {
            color: var(--text-secondary);
            font-size: 0.84rem;
        }

        /* Form styling */
        div[data-testid="stForm"] {
            background: rgba(15, 23, 42, 0.75) !important;
            backdrop-filter: blur(24px) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 24px !important;
            padding: 2.2rem !important;
            box-shadow: var(--card-shadow) !important;
        }

        /* Inputs & Controls override */
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
            background: rgba(30, 41, 59, 0.7) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 12px !important;
            color: #fff !important;
        }

        div[data-baseweb="input"] input, div[data-baseweb="select"] input {
            color: #fff !important;
        }

        .stSelectbox label, .stTextInput label, .stNumberInput label, .stSlider label {
            color: var(--text-secondary) !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
        }

        /* Submit Button */
        div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
            background: var(--primary-gradient) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 14px !important;
            font-weight: 800 !important;
            font-size: 1rem !important;
            padding: 0.75rem 1.6rem !important;
            box-shadow: 0 10px 28px rgba(6, 182, 212, 0.35) !important;
            transition: all 0.3s ease !important;
        }

        div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
            transform: translateY(-2px) scale(1.01) !important;
            box-shadow: 0 14px 38px rgba(6, 182, 212, 0.55) !important;
        }

        div.stDownloadButton > button {
            background: rgba(6, 182, 212, 0.15) !important;
            border: 1px solid rgba(6, 182, 212, 0.4) !important;
            color: #38bdf8 !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            padding: 0.65rem 1.2rem !important;
            box-shadow: 0 8px 20px rgba(6, 182, 212, 0.2) !important;
            transition: all 0.3s ease !important;
        }

        div.stDownloadButton > button:hover {
            background: rgba(6, 182, 212, 0.3) !important;
            color: #ffffff !important;
            transform: translateY(-2px) !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: rgba(11, 17, 32, 0.88) !important;
            backdrop-filter: blur(24px) !important;
            border-right: 1px solid var(--glass-border) !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--text-secondary) !important;
        }

        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] strong {
            color: #fff !important;
        }

        /* Streamlit Tabs Styling */
        button[data-baseweb="tab"] {
            color: var(--text-secondary) !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.2rem !important;
            border-radius: 10px !important;
        }

        button[aria-selected="true"] {
            color: var(--primary-cyan) !important;
            background: rgba(6, 182, 212, 0.12) !important;
            border-bottom: 2px solid var(--primary-cyan) !important;
        }

        .section-header {
            font-size: 1.8rem;
            font-weight: 800;
            color: #fff;
            letter-spacing: -0.03em;
            margin: 2rem 0 0.8rem;
        }

        .section-sub {
            color: var(--text-secondary);
            margin-bottom: 1.4rem;
            font-size: 0.95rem;
        }

        .footer-credit {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            padding-top: 2.5rem;
            border-top: 1px solid rgba(255,255,255,0.06);
            margin-top: 3rem;
        }
    </style>

    <!-- Dynamic Canvas Particle Grid & Floating Orbs Overlay -->
    <div class="moving-bg-overlay">
        <div class="bg-orb bg-orb-1"></div>
        <div class="bg-orb bg-orb-2"></div>
        <div class="bg-orb bg-orb-3"></div>
        <div class="bg-orb bg-orb-4"></div>
        <canvas id="bg-particle-canvas" style="position:absolute; width:100%; height:100%; top:0; left:0; opacity:0.4;"></canvas>
    </div>

    <script>
        (function() {
            const canvas = document.getElementById('bg-particle-canvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            let width = canvas.width = window.innerWidth;
            let height = canvas.height = window.innerHeight;

            window.addEventListener('resize', () => {
                width = canvas.width = window.innerWidth;
                height = canvas.height = window.innerHeight;
            });

            const particles = [];
            const count = 45;
            for (let i = 0; i < count; i++) {
                particles.push({
                    x: Math.random() * width,
                    y: Math.random() * height,
                    vx: (Math.random() - 0.5) * 0.45,
                    vy: (Math.random() - 0.5) * 0.45,
                    r: Math.random() * 2 + 1.2,
                    alpha: Math.random() * 0.5 + 0.2
                });
            }

            function draw() {
                ctx.clearRect(0, 0, width, height);
                for (let i = 0; i < particles.length; i++) {
                    let p = particles[i];
                    p.x += p.vx;
                    p.y += p.vy;

                    if (p.x < 0 || p.x > width) p.vx *= -1;
                    if (p.y < 0 || p.y > height) p.vy *= -1;

                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(6, 182, 212, ${p.alpha})`;
                    ctx.fill();

                    for (let j = i + 1; j < particles.length; j++) {
                        let p2 = particles[j];
                        let dx = p.x - p2.x;
                        let dy = p.y - p2.y;
                        let dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist < 140) {
                            ctx.beginPath();
                            ctx.moveTo(p.x, p.y);
                            ctx.lineTo(p2.x, p2.y);
                            ctx.strokeStyle = `rgba(6, 182, 212, ${(1 - dist / 140) * 0.2})`;
                            ctx.lineWidth = 0.7;
                            ctx.stroke();
                        }
                    }
                }
                requestAnimationFrame(draw);
            }
            draw();
        })();
    </script>
    """,
    unsafe_allow_html=True,
)

if "profile_name" not in st.session_state:
    st.session_state.profile_name = ""
if "profile_email" not in st.session_state:
    st.session_state.profile_email = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Swasthya Sakha Auth Screen Implementation
if not st.session_state.authenticated:
    auth_left, auth_right = st.columns([1.1, 0.9], gap="large")

    with auth_left:
        st.markdown(
            """
            <div class="auth-card-left">
                <div class="splash-logo-box">
                    <div class="splash-logo-glow"></div>
                    <div class="splash-logo-card">✦</div>
                </div>
                <div class="brand-badge" style="align-self: flex-start; margin-bottom: 0.6rem;">✦ ACADEMIC WELLBEING PLATFORM</div>
                <h1 class="auth-title">Your study journey deserves a clearer path.</h1>
                <p class="auth-subtitle">
                    Understand your exam outlook with AI precision, spot key study levers, and navigate each week with total confidence.
                </p>
                <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                    <div style="background: rgba(6, 182, 212, 0.1); border: 1px solid rgba(6, 182, 212, 0.25); border-radius: 12px; padding: 0.5rem 0.9rem; font-size: 0.82rem; color: #06b6d4; font-weight: 600;">
                        ✦ 500K Student Baseline
                    </div>
                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 12px; padding: 0.5rem 0.9rem; font-size: 0.82rem; color: #10b981; font-weight: 600;">
                        ⚡ 26 Academic Signals
                    </div>
                    <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 12px; padding: 0.5rem 0.9rem; font-size: 0.82rem; color: #a78bfa; font-weight: 600;">
                        🔒 Private & Local
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with auth_right:
        st.markdown(
            """
            <div class="auth-card-right">
                <div class="auth-form-header">
                    <h2>Welcome to StudyPulse</h2>
                    <p>Enter your profile to unlock your personalized prediction studio.</p>
                </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("access_form"):
            access_name = st.text_input("Your Name", value=st.session_state.profile_name, placeholder="e.g. Aanya Sharma")
            access_email = st.text_input("Email Address", value=st.session_state.profile_email, placeholder="student@university.edu")
            accepted = st.checkbox("I understand this is an AI academic forecast model", value=True)
            access = st.form_submit_button("Continue to Workspace  →", use_container_width=True)

        st.markdown(
            """
                <p style="color: var(--text-muted); font-size: 0.8rem; margin-top: 1rem; text-align: center;">
                    ⚡ No login password required. Session data remains private in your browser.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if access:
            if not access_name.strip() or "@" not in access_email or not accepted:
                st.error("Please enter a valid name, email address, and accept the disclaimer to enter.")
            else:
                st.session_state.profile_name = access_name.strip()
                st.session_state.profile_email = access_email.strip()
                st.session_state.authenticated = True
                st.rerun()

    st.stop()

# Header for Authenticated Workspace
st.markdown(
    f"""
    <div class="glass-header">
        <div class="brand-logo">
            <div class="brand-icon-box">✦</div>
            <span>StudyPulse</span>
        </div>
        <div style="display: flex; align-items: center; gap: 14px;">
            <span class="brand-badge">✦ AI INTELLIGENCE</span>
            <span style="color: var(--text-secondary); font-size: 0.88rem; font-weight: 600;">
                👋 Welcome, <strong style="color:#fff">{escape(st.session_state.profile_name)}</strong>
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Control
with st.sidebar:
    st.markdown("### ✦ Student Workspace")
    name = st.text_input("Student Name", value=st.session_state.profile_name, placeholder="e.g. Aanya Sharma")
    st.session_state.profile_name = name
    st.markdown("---")
    st.markdown("**How StudyPulse Works**")
    st.caption("1. Input academic history & daily wellbeing routine\n\n2. AI model predicts expected final exam performance\n\n3. Get actionable recommendations & shareable report")
    st.markdown("---")
    st.caption("Predictions are model estimates for support and study planning.")
    if st.button("Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.result = None
        st.rerun()


st.markdown("<div class='section-header'>Configure your input signals</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Select values representing your current study semester. Pre-filled with dataset averages.</div>", unsafe_allow_html=True)


def select_for(column, label):
    values = list(label_encoders[column].classes_)
    default = str(reference[column].mode().iloc[0])
    return st.selectbox(label, values, index=values.index(default) if default in values else 0)


with st.form("prediction_form"):
    academics, habits, background = st.tabs(["Academic Record", "Study & Wellbeing", "Background Context"])
    values = {}
    with academics:
        a1, a2, a3 = st.columns(3)
        values["Student_ID"] = a1.number_input("Student ID", min_value=1, value=250000, step=1)
        values["Semester_ID"] = a2.number_input("Semester", min_value=1, max_value=8, value=5, step=1)
        values["Age"] = a3.number_input("Age", min_value=17, max_value=30, value=23, step=1)
        values["Previous_GPA"] = a1.number_input("Previous GPA", min_value=0.0, max_value=4.0, value=2.80, step=0.01)
        values["Midterm_Mark"] = a2.number_input("Midterm Mark", min_value=0.0, max_value=100.0, value=53.6, step=0.1)
        values["Number_of_Failed_Courses"] = a3.number_input("Failed Courses", min_value=0, max_value=10, value=1, step=1)
        values["Total_Credits_Earned"] = a1.number_input("Credits Earned", min_value=0, max_value=200, value=68, step=1)
        values["Library_Visits_Per_Month"] = a2.number_input("Library Visits / Month", min_value=0, max_value=30, value=9, step=1)
        values["Major_Subject"] = a3.selectbox("Major Subject", list(label_encoders["Major_Subject"].classes_))
        values["University_Name"] = a1.selectbox("University Name", list(label_encoders["University_Name"].classes_))

    with habits:
        h1, h2, h3 = st.columns(3)
        values["Weekly_Study_Hours"] = h1.slider("Weekly Study Hours", 0.0, 45.0, 18.0, 0.5)
        values["Attendance_Rate"] = h2.slider("Attendance Rate (%)", 0.0, 100.0, 81.9, 0.1)
        values["Sleep_Hours"] = h3.slider("Sleep Hours / Night", 3.0, 11.0, 6.8, 0.1)
        values["Internet_Quality"] = h1.slider("Internet Quality", 1.0, 10.0, 6.5, 0.1)
        values["Study_Space_Quality"] = h2.slider("Study Space Quality", 1.0, 10.0, 6.5, 0.1)
        values["Stress_Level"] = h3.slider("Stress Level", 1.0, 10.0, 5.5, 0.1)
        values["Motivation_Score"] = h1.slider("Motivation Score", 1.0, 10.0, 6.0, 0.1)
        values["Self_Efficacy_Score"] = h2.slider("Self-Efficacy Score", 1.0, 10.0, 4.5, 0.1)
        values["Social_Media_Usage_Hours"] = h3.slider("Social Media Hours / Day", 0.0, 12.0, 2.1, 0.1)
        values["Extracurricular_Hours"] = h1.slider("Extracurricular Hours / Week", 0.0, 30.0, 2.8, 0.1)

    with background:
        b1, b2, b3 = st.columns(3)
        values["Gender"] = select_for("Gender", "Gender")
        values["Region_Type"] = select_for("Region_Type", "Region Type")
        values["Family_Size"] = b3.number_input("Family Size", min_value=1, max_value=15, value=5, step=1)
        values["Home_City"] = b1.selectbox("Home City", list(label_encoders["Home_City"].classes_))
        values["Parent_Education"] = b2.selectbox("Parent Education", list(label_encoders["Parent_Education"].classes_))
        values["Family_Income_Level"] = b3.selectbox("Family Income Level", list(label_encoders["Family_Income_Level"].classes_))

    submitted = st.form_submit_button("Generate Prediction  →", use_container_width=True)

if submitted:
    encoded = values.copy()
    for column, encoder in label_encoders.items():
        encoded[column] = encoder.transform([values[column]])[0]
    input_frame = pd.DataFrame([[encoded[column] for column in feature_columns]], columns=feature_columns)
    score = float(np.clip(model.predict(input_frame)[0], 0, 100))
    percentile = float((reference["Final_Exam_Score"] <= score).mean() * 100)
    st.session_state.result = {"score": score, "percentile": percentile, "values": values}
    
    # Append to prediction history CSV
    append_prediction_history(values["Student_ID"], score)

# Display Prediction Result Section (Replaces previous UI part as requested by user)
if st.session_state.result:
    result = st.session_state.result
    score, percentile, values = result["score"], result["percentile"], result["values"]
    
    band = (
        "Excellent Trajectory"
        if score >= 75
        else "Strong Foundation"
        if score >= 60
        else "Growth Opportunity"
        if score >= 45
        else "Needs Improvement"
    )

    st.markdown("<div class='section-header'>Prediction Result</div>", unsafe_allow_html=True)

    # 3 Stat Cards in a Row
    res_c1, res_c2, res_c3 = st.columns(3)
    with res_c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Predicted Final Score</div>
                <div class="stat-value">{score:.2f} / 100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with res_c2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Performance Level</div>
                <div class="stat-value" style="font-size:1.65rem;">{band}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with res_c3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Attendance Rate</div>
                <div class="stat-value">{values['Attendance_Rate']:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Two Side-by-Side Charts (Student Input Overview & Predicted Final Score Gauge)
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Bar Chart: Student Input Overview
        bar_data = pd.DataFrame({
            "Feature": ["Attendance Rate", "Midterm Mark", "Study Hours", "Sleep Hours", "Motivation"],
            "Value": [
                values.get("Attendance_Rate", 0),
                values.get("Midterm_Mark", 0),
                values.get("Weekly_Study_Hours", 0),
                values.get("Sleep_Hours", 0),
                values.get("Motivation_Score", 0)
            ]
        })

        fig_bar = go.Figure(data=[
            go.Bar(
                x=bar_data["Feature"],
                y=bar_data["Value"],
                marker_color=["#38bdf8", "#818cf8", "#6366f1", "#2dd4bf", "#fbbf24"],
                text=bar_data["Value"],
                textposition="auto"
            )
        ])
        fig_bar.update_layout(
            title={"text": "Student Input Overview", "font": {"size": 17, "color": "#f8fafc"}},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
            yaxis_title="Value",
            xaxis=dict(tickangle=-15),
            margin=dict(l=20, r=20, t=50, b=30),
            height=320
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        # Gauge Chart: Predicted Final Score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Predicted Final Score", 'font': {'size': 17, 'color': '#f8fafc'}},
            number={'suffix': "/100", 'font': {'size': 32, 'color': '#ffffff'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': "#ffffff", 'thickness': 0.15},
                'bgcolor': "rgba(0,0,0,0)",
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, 45], 'color': '#ef4444'},
                    {'range': [45, 60], 'color': '#f97316'},
                    {'range': [60, 75], 'color': '#eab308'},
                    {'range': [75, 100], 'color': '#10b981'}
                ]
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
            margin=dict(l=30, r=30, t=50, b=30),
            height=320
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Download Prediction Report Section with Format Choice (CSV, HTML, JPEG, PDF)
    dl_col1, dl_col2 = st.columns([1, 2])
    with dl_col1:
        report_fmt = st.selectbox(
            "Select report format:",
            options=["csv", "html", "jpeg", "pdf"],
            format_func=lambda x: f".{x.lower()}",
            key="report_format_selector"
        )
    
    rep_data, rep_fname, rep_mime = generate_report_file(
        values, score, band, report_fmt.upper(), profile_name=st.session_state.profile_name
    )
    
    with dl_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="⬇ Download This Prediction Report",
            data=rep_data,
            file_name=rep_fname,
            mime=rep_mime,
            use_container_width=True
        )

    st.markdown("---")

    # Prediction History Section
    st.markdown("<div class='section-header'>Prediction History</div>", unsafe_allow_html=True)
    
    history_df = load_prediction_history()
    
    if not history_df.empty:
        # Display history dataframe formatted like screenshot
        disp_df = history_df.copy()
        disp_df["Predicted Score"] = disp_df["Predicted Score"].apply(lambda x: f"{float(x):.4f}")
        st.dataframe(disp_df, use_container_width=True, height=180)
        
        hist_dl_col1, hist_dl_col2 = st.columns([1, 2])
        with hist_dl_col1:
            history_fmt = st.selectbox(
                "Select history format:",
                options=["csv", "html", "jpeg", "pdf"],
                format_func=lambda x: f".{x.lower()}",
                key="history_format_selector"
            )
            
        hist_data, hist_fname, hist_mime = generate_history_file(history_df, history_fmt.upper())
        
        with hist_dl_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="⬇ Download Prediction History",
                data=hist_data,
                file_name=hist_fname,
                mime=hist_mime,
                use_container_width=True
            )
    else:
        st.caption("No prediction history recorded yet.")

st.markdown("<div class='footer-credit'>✦ StudyPulse · AI Academic Performance & Wellbeing Intelligence</div>", unsafe_allow_html=True)
