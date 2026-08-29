import streamlit as st
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import soundfile as sf
import tensorflow as tf
import librosa
from audio_utils import extract_mfcc_from_audio, LANGUAGES
from auth_utils import authenticate, register_user, load_users

# Page Configuration
st.set_page_config(
    page_title="LingAI | Multilingual Spoken Language Identification",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_info" not in st.session_state:
    st.session_state["user_info"] = None
if "theme" not in st.session_state:
    st.session_state["theme"] = "🌙 Dark Mode"

def get_theme_css(is_dark):
    if is_dark:
        return """
        <style>
            .stApp {
                background-color: #080c16 !important;
                color: #f8fafc !important;
            }
            p, span, label, div { color: #e2e8f0; }
            h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-family: 'Outfit', sans-serif !important; }
            section[data-testid="stSidebar"] {
                background-color: #0b1120 !important;
                border-right: 1px solid #1e293b !important;
            }
            section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
            .hero-container {
                background: linear-gradient(135deg, #131d38 0%, #0c1222 100%);
                border: 1px solid #2563eb;
                border-radius: 18px;
                padding: 24px 30px;
                margin-bottom: 22px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            }
            .hero-badge {
                display: inline-block;
                background: #0369a1;
                color: #e0f2fe !important;
                padding: 4px 12px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
                margin-bottom: 8px;
            }
            .hero-title {
                font-size: 30px !important;
                font-weight: 800 !important;
                color: #ffffff !important;
                margin: 0 0 6px 0;
            }
            .hero-subtitle {
                color: #cbd5e1 !important;
                font-size: 14px !important;
                line-height: 1.5;
            }
            .section-title {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 18px;
                font-weight: 700;
                color: #38bdf8 !important;
                margin-bottom: 12px;
            }
            .result-card {
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border: 2px solid #38bdf8;
                border-radius: 16px;
                padding: 22px;
                text-align: center;
                box-shadow: 0 8px 25px rgba(56, 189, 248, 0.15);
                margin-bottom: 16px;
            }
            .result-card-header {
                font-size: 11px;
                font-weight: 700;
                color: #94a3b8 !important;
                text-transform: uppercase;
                letter-spacing: 0.1em;
            }
            .result-card-title {
                font-size: 32px !important;
                font-weight: 800 !important;
                color: #38bdf8 !important;
                margin: 6px 0 2px 0;
            }
            .result-card-subtitle {
                font-size: 15px;
                color: #cbd5e1 !important;
                margin-bottom: 10px;
            }
            .confidence-badge {
                display: inline-block;
                background: #059669;
                color: #ffffff !important;
                padding: 5px 16px;
                border-radius: 999px;
                font-size: 16px;
                font-weight: 700;
            }
            .transcript-card {
                background: #111827;
                border-left: 4px solid #38bdf8;
                border-radius: 0 10px 10px 0;
                padding: 12px 16px;
                margin: 10px 0 14px 0;
            }
            .chip {
                display: inline-block;
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 600;
                color: #f1f5f9 !important;
                margin: 2px;
            }
            .stat-card {
                background: #111827;
                border: 1px solid #1e293b;
                border-radius: 12px;
                padding: 14px;
                text-align: center;
            }
            .stat-val {
                font-size: 24px;
                font-weight: 800;
                color: #38bdf8 !important;
            }
            .stat-lbl {
                font-size: 11px;
                color: #94a3b8 !important;
                text-transform: uppercase;
                font-weight: 600;
            }
            .auth-container {
                max-width: 480px;
                margin: 40px auto;
                background: linear-gradient(135deg, rgba(19, 29, 56, 0.95) 0%, rgba(12, 18, 34, 0.98) 100%);
                border: 1px solid rgba(59, 130, 246, 0.4);
                border-radius: 24px;
                padding: 36px 32px;
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7);
                text-align: center;
            }
            .creds-box {
                background: rgba(30, 41, 59, 0.6);
                border: 1px dashed #475569;
                border-radius: 10px;
                padding: 12px 16px;
                margin-top: 18px;
                font-size: 12px;
                color: #94a3b8;
                text-align: left;
            }
            .user-profile-card {
                background: #111827;
                border: 1px solid #1e293b;
                border-radius: 12px;
                padding: 12px 14px;
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 15px;
            }
            .panel-box {
                background: #111827;
                border: 1px solid #1e293b;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 14px;
            }
        </style>
        """
    else:
        return """
        <style>
            .stApp {
                background-color: #f8fafc !important;
                color: #0f172a !important;
            }
            p, span, label, div { color: #1e293b; }
            h1, h2, h3, h4, h5, h6 { color: #0f172a !important; font-family: 'Outfit', sans-serif !important; }
            section[data-testid="stSidebar"] {
                background-color: #f1f5f9 !important;
                border-right: 1px solid #e2e8f0 !important;
            }
            section[data-testid="stSidebar"] * { color: #0f172a !important; }
            .hero-container {
                background: linear-gradient(135deg, #e0f2fe 0%, #eff6ff 100%);
                border: 1px solid #93c5fd;
                border-radius: 18px;
                padding: 24px 30px;
                margin-bottom: 22px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            }
            .hero-badge {
                display: inline-block;
                background: #0284c7;
                color: #ffffff !important;
                padding: 4px 12px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
                margin-bottom: 8px;
            }
            .hero-title {
                font-size: 30px !important;
                font-weight: 800 !important;
                color: #0369a1 !important;
                margin: 0 0 6px 0;
            }
            .hero-subtitle {
                color: #334155 !important;
                font-size: 14px !important;
                line-height: 1.5;
            }
            .section-title {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 18px;
                font-weight: 700;
                color: #0284c7 !important;
                margin-bottom: 12px;
            }
            .result-card {
                background: linear-gradient(135deg, #f0fdf4 0%, #ecfeff 100%);
                border: 2px solid #0284c7;
                border-radius: 16px;
                padding: 22px;
                text-align: center;
                box-shadow: 0 6px 20px rgba(2, 132, 199, 0.1);
                margin-bottom: 16px;
            }
            .result-card-header {
                font-size: 11px;
                font-weight: 700;
                color: #64748b !important;
                text-transform: uppercase;
                letter-spacing: 0.1em;
            }
            .result-card-title {
                font-size: 32px !important;
                font-weight: 800 !important;
                color: #0284c7 !important;
                margin: 6px 0 2px 0;
            }
            .result-card-subtitle {
                font-size: 15px;
                color: #475569 !important;
                margin-bottom: 10px;
            }
            .confidence-badge {
                display: inline-block;
                background: #10b981;
                color: #ffffff !important;
                padding: 5px 16px;
                border-radius: 999px;
                font-size: 16px;
                font-weight: 700;
            }
            .transcript-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-left: 4px solid #0284c7;
                border-radius: 0 10px 10px 0;
                padding: 12px 16px;
                margin: 10px 0 14px 0;
            }
            .chip {
                display: inline-block;
                background: #e2e8f0;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 600;
                color: #1e293b !important;
                margin: 2px;
            }
            .stat-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 14px;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.03);
            }
            .stat-val {
                font-size: 24px;
                font-weight: 800;
                color: #0284c7 !important;
            }
            .stat-lbl {
                font-size: 11px;
                color: #64748b !important;
                text-transform: uppercase;
                font-weight: 600;
            }
            .auth-container {
                max-width: 480px;
                margin: 40px auto;
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 24px;
                padding: 36px 32px;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
                text-align: center;
            }
            .creds-box {
                background: #f8fafc;
                border: 1px dashed #cbd5e1;
                border-radius: 10px;
                padding: 12px 16px;
                margin-top: 18px;
                font-size: 12px;
                color: #475569;
                text-align: left;
            }
            .user-profile-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 12px 14px;
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 15px;
            }
            .panel-box {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 14px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            }
        </style>
        """

# 14 Languages Metadata
LANG_METADATA = {
    "English": {
        "flag": "🇬🇧", "native": "English", "family": "Germanic", "region": "Global / Official",
        "color": "#0284c7", "sample_text": "Spoken language identification is a key technology for speech recognition and translation.",
        "script": "Latin Script"
    },
    "Hindi": {
        "flag": "🇮🇳", "native": "हिन्दी", "family": "Indo-Aryan", "region": "North & Central India",
        "color": "#ea580c", "sample_text": "नमस्ते, यह भाषा पहचान प्रणाली का एक वास्तविक भाषण नमूना है।",
        "script": "Devanagari Script"
    },
    "Mandarin": {
        "flag": "🇨🇳", "native": "中文 (普通话)", "family": "Sino-Tibetan", "region": "East Asia",
        "color": "#dc2626", "sample_text": "你好，这是一个用于语音语言识别的真实中文普通话样本。",
        "script": "Simplified Chinese"
    },
    "Tamil": {
        "flag": "🇮🇳", "native": "தமிழ்", "family": "Dravidian", "region": "Tamil Nadu / South India",
        "color": "#9333ea", "sample_text": "வணக்கம், இது மொழி அடையாளம் காண்பதற்கான உண்மையான தமிழ் பேச்சு மாதிரி ஆகும்.",
        "script": "Tamil Script"
    },
    "Telugu": {
        "flag": "🇮🇳", "native": "తెలుగు", "family": "Dravidian", "region": "Andhra & Telangana",
        "color": "#2563eb", "sample_text": "నమస్కారం, ఇది భాష గుర్తింపు కోసం నిజమైన తెలుగు మాట్లాడే నమూనా.",
        "script": "Telugu Script"
    },
    "Kannada": {
        "flag": "🇮🇳", "native": "ಕನ್ನಡ", "family": "Dravidian", "region": "Karnataka / South India",
        "color": "#ca8a04", "sample_text": "ನಮಸ್ಕಾರ, ಇದು ಭಾಷಾ ಗುರುತಿಸುವಿಕೆಗಾಗಿ ನೈಜ ಕನ್ನಡ ಭಾಷಣದ ಧ್ವನಿ ಮಾದರಿಯಾಗಿದೆ.",
        "script": "Kannada Script"
    },
    "Malayalam": {
        "flag": "🇮🇳", "native": "മലയാളം", "family": "Dravidian", "region": "Kerala / South India",
        "color": "#059669", "sample_text": "നമസ്കാരം, ഇത് ഭാഷ തിരിച്ചറിയുന്നതിനുള്ള യഥാർത്ഥ മലയാള സംഭാഷണ സാമ്പിളാണ്.",
        "script": "Malayalam Script"
    },
    "Bengali": {
        "flag": "🇮🇳", "native": "বাংলা", "family": "Indo-Aryan", "region": "West Bengal / East India",
        "color": "#db2777", "sample_text": "নমস্কার, এটি ভাষা শনাক্তকরণের জন্য একটি প্রকৃত বাংলা কথ্য অডিও নমুনা।",
        "script": "Bengali Script"
    },
    "Marathi": {
        "flag": "🇮🇳", "native": "मराठी", "family": "Indo-Aryan", "region": "Maharashtra / West India",
        "color": "#d97706", "sample_text": "नमस्कार, हा भाषा ओळखण्यासाठी एक खरा मराठी बोलण्याचा ऑडिओ नमुना आहे.",
        "script": "Devanagari Script"
    },
    "Gujarati": {
        "flag": "🇮🇳", "native": "ગુજરાતી", "family": "Indo-Aryan", "region": "Gujarat / West India",
        "color": "#0d9488", "sample_text": "નમસ્તે, આ ભાષા ઓળખ માટેનો એક અસલી ગુજરાતી બોલવાનો નમૂનો છે.",
        "script": "Gujarati Script"
    },
    "Punjabi": {
        "flag": "🇮🇳", "native": "ਪੰਜਾਬੀ", "family": "Indo-Aryan", "region": "Punjab / North India",
        "color": "#eab308", "sample_text": "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ, ਇਹ ਭਾਸ਼ਾ ਪਛਾਣ ਲਈ ਇੱਕ ਅਸਲੀ ਪੰਜਾਬੀ ਬੋਲਣ ਵਾਲਾ ਨਮੂਨਾ ਹੈ।",
        "script": "Gurmukhi Script"
    },
    "French": {
        "flag": "🇫🇷", "native": "Français", "family": "Romance", "region": "France & Western Europe",
        "color": "#3b82f6", "sample_text": "Bonjour, ceci est un enregistrement vocal authentique pour l'identification de la langue parlée.",
        "script": "Latin Script"
    },
    "German": {
        "flag": "🇩🇪", "native": "Deutsch", "family": "Germanic", "region": "Germany & Central Europe",
        "color": "#f59e0b", "sample_text": "Guten Tag, dies ist eine authentische Sprachaufnahme zur automatischen Erkennung gesprochener Sprachen.",
        "script": "Latin Script"
    },
    "Spanish": {
        "flag": "🇪🇸", "native": "Español", "family": "Romance", "region": "Spain & Latin America",
        "color": "#ef4444", "sample_text": "Hola, esta es una muestra de audio auténtica para la identificación automática del idioma hablado.",
        "script": "Latin Script"
    }
}

# Reliable model loader without stale cache lock
def load_trained_model():
    if "cached_model" in st.session_state and st.session_state["cached_model"] is not None:
        if st.session_state["cached_model"].output_shape[-1] == len(LANGUAGES):
            return st.session_state["cached_model"]
            
    model_path = "sld.keras"
    if os.path.exists(model_path):
        try:
            m = tf.keras.models.load_model(model_path)
            if m.output_shape[-1] == len(LANGUAGES):
                st.session_state["cached_model"] = m
                return m
        except Exception as e:
            print("Model load error:", e)
            return None
    return None

# -----------------------------------------------------------------
# LOGIN / REGISTRATION VIEW
# -----------------------------------------------------------------
def render_login(is_dark):
    col_l, col_center, col_r = st.columns([1, 1.8, 1])
    
    with col_center:
        st.markdown("""
        <div class="auth-container">
            <div style="font-size: 40px; margin-bottom: 8px;">🎙️</div>
            <div style="font-family: 'Outfit'; font-size: 28px; font-weight: 800; margin-bottom: 4px;">LingAI Platform</div>
            <div style="font-size: 14px; opacity: 0.8; margin-bottom: 20px;">Neural Spoken Language Identification System</div>
        </div>
        """, unsafe_allow_html=True)
        
        tab_login, tab_signup, tab_demo = st.tabs(["🔑 Sign In", "📝 Create Account", "⚡ Quick Demo"])
        
        with tab_login:
            with st.form("login_form"):
                st.markdown("<div style='font-size: 14px; font-weight: 600; margin-bottom: 4px;'>Username or Email:</div>", unsafe_allow_html=True)
                username_input = st.text_input("Username", placeholder="e.g. admin or demo", label_visibility="collapsed")
                
                st.markdown("<div style='font-size: 14px; font-weight: 600; margin: 10px 0 4px 0;'>Password:</div>", unsafe_allow_html=True)
                password_input = st.text_input("Password", type="password", placeholder="Enter your password", label_visibility="collapsed")
                
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                submit_login = st.form_submit_button("🚀 Sign In to Dashboard", use_container_width=True, type="primary")
                
                if submit_login:
                    if not username_input or not password_input:
                        st.error("Please enter both username and password.")
                    else:
                        success, user_data = authenticate(username_input, password_input)
                        if success:
                            st.session_state["authenticated"] = True
                            st.session_state["user_info"] = user_data
                            st.success(f"Welcome back, {user_data['name']}!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")

            st.markdown("""
            <div class="creds-box">
                <b>💡 Demo Credentials:</b><br>
                • <b>Username:</b> <code>admin</code> | <b>Password:</b> <code>admin123</code><br>
                • <b>Username:</b> <code>demo</code>  | <b>Password:</b> <code>demo123</code>
            </div>
            """, unsafe_allow_html=True)

        with tab_signup:
            with st.form("signup_form"):
                st.markdown("<div style='font-size: 13px; font-weight: 600;'>Full Name:</div>", unsafe_allow_html=True)
                new_name = st.text_input("Full Name", placeholder="e.g. Dr. Priya Sharma", label_visibility="collapsed")
                
                st.markdown("<div style='font-size: 13px; font-weight: 600; margin-top: 6px;'>Username:</div>", unsafe_allow_html=True)
                new_username = st.text_input("New Username", placeholder="e.g. psharma", label_visibility="collapsed")
                
                st.markdown("<div style='font-size: 13px; font-weight: 600; margin-top: 6px;'>Email Address:</div>", unsafe_allow_html=True)
                new_email = st.text_input("Email", placeholder="e.g. priya@university.edu", label_visibility="collapsed")
                
                st.markdown("<div style='font-size: 13px; font-weight: 600; margin-top: 6px;'>Password:</div>", unsafe_allow_html=True)
                new_password = st.text_input("New Password", type="password", placeholder="Choose a secure password", label_visibility="collapsed")
                
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                submit_signup = st.form_submit_button("✨ Register Account", use_container_width=True, type="primary")
                
                if submit_signup:
                    success, msg = register_user(new_username, new_name, new_email, new_password)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

        with tab_demo:
            st.markdown("<div style='padding: 10px 0;'>Click below to access the full platform instantly as a guest researcher:</div>", unsafe_allow_html=True)
            if st.button("⚡ Continue with Instant Demo Access", use_container_width=True, type="primary"):
                st.session_state["authenticated"] = True
                st.session_state["user_info"] = {
                    "name": "Guest Researcher",
                    "role": "Audio Analyst",
                    "email": "guest@lingai.org"
                }
                st.rerun()

# -----------------------------------------------------------------
# MAIN DASHBOARD
# -----------------------------------------------------------------
def render_dashboard(is_dark):
    user = st.session_state["user_info"] or {"name": "Researcher", "role": "Audio Analyst"}
    
    # SIDEBAR
    with st.sidebar:
        # User Profile
        st.markdown(f"""
        <div class="user-profile-card">
            <div style="width: 38px; height: 38px; background: linear-gradient(135deg, #0284c7, #6366f1); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px; color: #ffffff; font-weight: 700;">{user['name'][0].upper()}</div>
            <div style="flex-grow: 1; overflow: hidden;">
                <div style="font-weight: 700; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{user['name']}</div>
                <div style="font-size: 11px; opacity: 0.8;">{user['role']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["user_info"] = None
            st.session_state["cached_model"] = None
            st.rerun()
            
        st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)
        
        # Navigation Menu
        menu = st.radio(
            "Navigation Menu",
            ["⚡ Real-Time Classifier", "🎧 Language Audio Gallery", "📊 Acoustic Spectrograms", "🏛️ Linguistic Families", "⚙️ Model Engine"]
        )
        
        st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)
        
        # Theme Toggle in Sidebar
        st.markdown("<div style='font-size: 12px; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;'>🎨 Theme Mode</div>", unsafe_allow_html=True)
        theme_choice = st.radio(
            "Theme Mode",
            ["🌙 Dark Mode", "☀️ Light Mode"],
            index=0 if st.session_state["theme"] == "🌙 Dark Mode" else 1,
            label_visibility="collapsed",
            horizontal=True
        )
        if theme_choice != st.session_state["theme"]:
            st.session_state["theme"] = theme_choice
            st.rerun()
        
        st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 12px; font-weight: 700; text-transform: uppercase; margin-bottom: 8px;'>Supported Languages (14)</div>", unsafe_allow_html=True)
        
        chips_html = "<div>"
        for lang in LANGUAGES:
            meta = LANG_METADATA[lang]
            chips_html += f"<span class='chip'>{meta['flag']} <b>{lang}</b></span>"
        chips_html += "</div>"
        st.markdown(chips_html, unsafe_allow_html=True)

    model = load_trained_model()

    # PAGE 1: REAL-TIME CLASSIFIER
    if menu == "⚡ Real-Time Classifier":
        st.markdown("""
        <div class="hero-container">
            <span class="hero-badge">⚡ Real-Time Speech Identification</span>
            <div class="hero-title">Spoken Language Classifier</div>
            <div class="hero-subtitle">Classify real human speech into <b>14 languages</b> across Dravidian, Indo-Aryan, Romance, Germanic, and Sino-Tibetan families using 64-dimensional Mel-Frequency Cepstral Coefficients (MFCC) and Deep LSTM Networks.</div>
        </div>
        """, unsafe_allow_html=True)
        
        if model is None:
            st.error("⚠️ Trained model is loading or missing. Please click the button below to train or reload:")
            if st.button("🚀 Train & Load Model Now", type="primary"):
                with st.spinner("Training model..."):
                    import subprocess
                    subprocess.run(["python", "language_identification.py"], capture_output=True, text=True)
                    st.session_state["cached_model"] = None
                    st.rerun()
            return

        col_input, col_result = st.columns([1.1, 0.9], gap="large")
        
        with col_input:
            st.markdown("<div class='section-title'>🎵 1. Choose Audio Input</div>", unsafe_allow_html=True)
            
            input_mode = st.radio(
                "Select Input Type:",
                ["🌟 Select Native Speech Sample", "🎙️ Record Microphone Voice", "📁 Upload Audio File (WAV/MP3)"]
            )
            
            audio_path = None
            
            if input_mode == "🌟 Select Native Speech Sample":
                sample_options = [f"sample_{l.lower()}.wav" for l in LANGUAGES]
                selected_sample = st.selectbox(
                    "Choose Language Sample to Test:",
                    sample_options,
                    index=0,
                    format_func=lambda x: f"{LANG_METADATA[x.replace('sample_', '').replace('.wav', '').capitalize()]['flag']} {x.replace('sample_', '').replace('.wav', '').capitalize()} ({LANG_METADATA[x.replace('sample_', '').replace('.wav', '').capitalize()]['native']})"
                )
                audio_path = os.path.join("./sample_audio", selected_sample)
                
                lang_key = selected_sample.replace('sample_', '').replace('.wav', '').capitalize()
                meta = LANG_METADATA[lang_key]
                st.markdown(f"""
                <div class="transcript-card">
                    <div style="font-size: 11px; font-weight: 700; color: #0284c7; text-transform: uppercase;">Spoken Text in {lang_key} ({meta['script']}):</div>
                    <div style="font-size: 15px; font-weight: 600; margin-top: 4px; line-height: 1.5;">"{meta['sample_text']}"</div>
                    <div style="font-size: 12px; opacity: 0.8; margin-top: 6px;">Linguistic Family: <b>{meta['family']}</b> • Region: <b>{meta['region']}</b></div>
                </div>
                """, unsafe_allow_html=True)
                
            elif input_mode == "🎙️ Record Microphone Voice":
                st.markdown("<div style='font-size: 14px; margin-bottom: 8px;'>Click the microphone button below to record your voice:</div>", unsafe_allow_html=True)
                mic_audio = st.audio_input("Record your voice")
                if mic_audio is not None:
                    os.makedirs("./temp_uploads", exist_ok=True)
                    audio_path = os.path.join("./temp_uploads", "mic_recording.wav")
                    with open(audio_path, "wb") as f:
                        f.write(mic_audio.getbuffer())
                    st.success("Audio captured successfully!")
                    
            else:
                uploaded_file = st.file_uploader("Upload .WAV or .MP3 audio file", type=["wav", "mp3"])
                if uploaded_file is not None:
                    os.makedirs("./temp_uploads", exist_ok=True)
                    audio_path = os.path.join("./temp_uploads", uploaded_file.name)
                    with open(audio_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
            
            if audio_path and os.path.exists(audio_path):
                st.markdown("<div style='font-size: 13px; font-weight: 600; margin-top: 8px;'>Audio Playback:</div>", unsafe_allow_html=True)
                st.audio(audio_path, format="audio/wav")
                
                with st.spinner("Extracting 64 MFCC features..."):
                    mfcc_feat = extract_mfcc_from_audio(audio_path)
                
                input_tensor = np.expand_dims(mfcc_feat, axis=0)
                preds = model.predict(input_tensor, verbose=0)[0]
                
                avg_prob = np.mean(preds, axis=0)
                predicted_class_idx = int(np.argmax(avg_prob))
                predicted_lang = LANGUAGES[predicted_class_idx]
                confidence = avg_prob[predicted_class_idx] * 100
                res_meta = LANG_METADATA[predicted_lang]

        with col_result:
            st.markdown("<div class='section-title'>🎯 2. Identification Result</div>", unsafe_allow_html=True)
            
            if audio_path and os.path.exists(audio_path):
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-card-header">Classified Language</div>
                    <div class="result-card-title">{res_meta['flag']} {predicted_lang}</div>
                    <div class="result-card-subtitle">{res_meta['native']} • {res_meta['family']} Family</div>
                    <div>
                        <span class="confidence-badge">{confidence:.1f}% Confidence</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<div style='font-size: 14px; font-weight: 700; margin-bottom: 8px;'>Probability Breakdown (Ranked)</div>", unsafe_allow_html=True)
                
                sorted_indices = np.argsort(avg_prob)[::-1]
                for idx in sorted_indices:
                    lang_item = LANGUAGES[idx]
                    p = avg_prob[idx]
                    meta_item = LANG_METADATA[lang_item]
                    
                    c_lbl, c_bar = st.columns([1.3, 3.7])
                    with c_lbl:
                        st.markdown(f"<span style='font-size: 13px; font-weight: 700; color: {meta_item['color']}'>{meta_item['flag']} {lang_item}</span>", unsafe_allow_html=True)
                    with c_bar:
                        st.progress(float(p))
                        st.caption(f"{p*100:.2f}%")
            else:
                st.markdown("""
                <div class="stat-card" style="padding: 45px 20px;">
                    <div style="font-size: 40px; margin-bottom: 8px;">🎙️</div>
                    <div style="font-size: 17px; font-weight: 700;">Awaiting Speech Input</div>
                    <div style="font-size: 13px; opacity: 0.8; margin-top: 4px;">Choose a language sample, record your voice, or upload a file.</div>
                </div>
                """, unsafe_allow_html=True)

        if audio_path and os.path.exists(audio_path):
            st.markdown("<hr style='opacity: 0.2; margin: 22px 0;'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>📊 3. 64-Channel MFCC Feature Spectrogram</div>", unsafe_allow_html=True)
            
            fig_bg = '#0b1120' if is_dark else '#ffffff'
            txt_c = '#cbd5e1' if is_dark else '#334155'
            spine_c = '#334155' if is_dark else '#cbd5e1'
            
            fig, ax = plt.subplots(figsize=(12, 3.2), facecolor=fig_bg)
            ax.set_facecolor(fig_bg)
            c = ax.imshow(mfcc_feat.T, aspect='auto', origin='lower', cmap='plasma')
            ax.set_title("Speech Formant Energy Trajectories (10 Seconds, 1,000 Frames)", color=txt_c, fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel("Time Frames (10ms steps)", color=txt_c, fontsize=10)
            ax.set_ylabel("MFCC Bins (0 - 63)", color=txt_c, fontsize=10)
            ax.tick_params(colors=txt_c, labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(spine_c)
            cb = fig.colorbar(c, ax=ax, fraction=0.015, pad=0.02)
            cb.ax.tick_params(colors=txt_c, labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)

    # PAGE 2: AUDIO GALLERY
    elif menu == "🎧 Language Audio Gallery":
        st.markdown("""
        <div class="hero-container">
            <span class="hero-badge">🎧 Audio Showcase</span>
            <div class="hero-title">Multilingual Speech Gallery</div>
            <div class="hero-subtitle">Listen to authentic spoken human sentences across all 14 supported languages with native scripts and regional metadata.</div>
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(2, gap="medium")
        for i, lang_name in enumerate(LANGUAGES):
            meta = LANG_METADATA[lang_name]
            sample_file = os.path.join("./sample_audio", f"sample_{lang_name.lower()}.wav")
            
            with cols[i % 2]:
                st.markdown(f"""
                <div class="panel-box" style="border-top: 3px solid {meta['color']};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <div style="font-size: 20px; font-weight: 800;">{meta['flag']} {lang_name} <span style="font-size: 15px; color: {meta['color']}; font-weight: 600;">({meta['native']})</span></div>
                        <span class="chip">{meta['family']}</span>
                    </div>
                    <div style="font-size: 12px; opacity: 0.8; margin-bottom: 10px;">Region: <b>{meta['region']}</b> • Script: <b>{meta['script']}</b></div>
                    <div class="transcript-card" style="margin: 6px 0 12px 0; font-size: 14px;">
                        "{meta['sample_text']}"
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists(sample_file):
                    st.audio(sample_file, format="audio/wav")

    # PAGE 3: SPECTROGRAMS
    elif menu == "📊 Acoustic Spectrograms":
        st.markdown("""
        <div class="hero-container">
            <span class="hero-badge">📊 Acoustic Profiling</span>
            <div class="hero-title">Comparative Spectrogram Explorer</div>
            <div class="hero-subtitle">Compare 64-dimensional Mel-Frequency Cepstral Coefficient patterns across Dravidian, Indo-Aryan, Romance, Germanic, and Sino-Tibetan linguistic families.</div>
        </div>
        """, unsafe_allow_html=True)
        
        if os.path.exists("mfcc_dataset.hdf5"):
            with h5py.File("mfcc_dataset.hdf5", 'r') as hf:
                X_train = hf['X_train'][:]
                Y_train = hf['Y_train'][:]
                X_val = hf['X_val'][:]
                Y_val = hf['Y_val'][:]
                
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='stat-card'><div class='stat-val'>{len(LANGUAGES)}</div><div class='stat-lbl'>Languages</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='stat-card'><div class='stat-val'>{len(X_train)}</div><div class='stat-lbl'>Train Sequences</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='stat-card'><div class='stat-val'>{len(X_val)}</div><div class='stat-lbl'>Validation Sequences</div></div>", unsafe_allow_html=True)
            c4.markdown("<div class='stat-card'><div class='stat-val'>64 × 1000</div><div class='stat-lbl'>MFCC Tensor</div></div>", unsafe_allow_html=True)
            
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Compare Selected Language Spectrograms</div>", unsafe_allow_html=True)
            selected_langs = st.multiselect("Select Languages to compare:", LANGUAGES, default=["Tamil", "French", "German", "Spanish"])
            
            if selected_langs:
                cols = st.columns(len(selected_langs))
                fig_bg = '#0b1120' if is_dark else '#ffffff'
                txt_c = '#cbd5e1' if is_dark else '#334155'
                spine_c = '#334155' if is_dark else '#cbd5e1'
                
                for col, lang_name in zip(cols, selected_langs):
                    lang_idx = LANGUAGES.index(lang_name)
                    mask = (np.argmax(Y_train[:, 0, :], axis=1) == lang_idx)
                    sample_feat = X_train[mask][0] if np.any(mask) else X_train[lang_idx]
                    
                    with col:
                        fig, ax = plt.subplots(figsize=(4, 4), facecolor=fig_bg)
                        ax.set_facecolor(fig_bg)
                        ax.imshow(sample_feat.T, aspect='auto', origin='lower', cmap='viridis')
                        ax.set_title(f"{LANG_METADATA[lang_name]['flag']} {lang_name}\n({LANG_METADATA[lang_name]['native']})", color=txt_c, fontsize=12, fontweight='bold')
                        ax.tick_params(colors=txt_c, labelsize=8)
                        for spine in ax.spines.values():
                            spine.set_color(spine_c)
                        plt.tight_layout()
                        st.pyplot(fig)

    # PAGE 4: LINGUISTIC FAMILIES
    elif menu == "🏛️ Linguistic Families":
        st.markdown("""
        <div class="hero-container">
            <span class="hero-badge">🏛️ Linguistic Taxonomy</span>
            <div class="hero-title">Indian & Global Linguistic Heritage</div>
            <div class="hero-subtitle">Explore the phonetic, structural, and acoustic traits of the 14 languages modeled across 5 major linguistic families.</div>
        </div>
        """, unsafe_allow_html=True)
        
        f1, f2 = st.columns(2, gap="medium")
        
        with f1:
            st.markdown("""
            <div class="panel-box" style="border-top: 4px solid #c084fc;">
                <h3 style="color: #c084fc !important; margin-top: 0;">🌴 Dravidian Family</h3>
                <p style="font-size: 13px; opacity: 0.85;">Spoken across South India. Characterized by agglutinative grammar, distinct retroflex consonants, and rich vowel harmony.</p>
                <div style="margin-top: 12px;">
                    <div style="margin-bottom: 8px;"><b>🇮🇳 Tamil (தமிழ்)</b><br><span style="font-size: 12px; opacity: 0.75;">Classical antiquity, alveolar stops, retroflex consonants.</span></div>
                    <div style="margin-bottom: 8px;"><b>🇮🇳 Telugu (తెలుగు)</b><br><span style="font-size: 12px; opacity: 0.75;">"Italian of the East" - melodic vowel endings.</span></div>
                    <div style="margin-bottom: 8px;"><b>🇮🇳 Kannada (ಕನ್ನಡ)</b><br><span style="font-size: 12px; opacity: 0.75;">Geminate consonant clusters, nasalized vowels.</span></div>
                    <div style="margin-bottom: 8px;"><b>🇮🇳 Malayalam (മലയാളം)</b><br><span style="font-size: 12px; opacity: 0.75;">Complex liquid sounds, dense consonant flow.</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="panel-box" style="border-top: 4px solid #38bdf8;">
                <h3 style="color: #38bdf8 !important; margin-top: 0;">🥐 Romance Family</h3>
                <p style="font-size: 13px; opacity: 0.85;">Descendants of Vulgar Latin spoken widely across Europe and the Americas. Renowned for nasal vowels, syllable-timed rhythm, and rich morphology.</p>
                <div style="margin-top: 12px;">
                    <div style="margin-bottom: 8px;"><b>🇫🇷 French (Français)</b><br><span style="font-size: 12px; opacity: 0.75;">Uvular fricative /r/, nasal vowel formants, non-phonemic stress.</span></div>
                    <div style="margin-bottom: 8px;"><b>🇪🇸 Spanish (Español)</b><br><span style="font-size: 12px; opacity: 0.75;">Pure 5-vowel system, rolled rhotics, clear syllable cadence.</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with f2:
            st.markdown("""
            <div class="panel-box" style="border-top: 4px solid #fb923c;">
                <h3 style="color: #fb923c !important; margin-top: 0;">🏔️ Indo-Aryan Family</h3>
                <p style="font-size: 13px; opacity: 0.85;">Branch of the Indo-European family spoken in North, Central, East, and West India. Characterized by aspiration contrasts.</p>
                <div style="margin-top: 12px;">
                    <div style="margin-bottom: 8px;"><b>🇮🇳 Hindi (हिन्दी)</b><br><span style="font-size: 12px; opacity: 0.75;">Syllable-timed cadence, aspirated stops.</span></div>
                    <div style="margin-bottom: 8px;"><b>🇮🇳 Bengali (বাংলা)</b><br><span style="font-size: 12px; opacity: 0.75;">Rounded vowel formant shifts, soft affricates.</span></div>
                    <div style="margin-bottom: 8px;"><b>🇮🇳 Marathi (मराठी)</b><br><span style="font-size: 12px; opacity: 0.75;">Retroflex lateral flap, dental/alveolar fricatives.</span></div>
                    <div style="margin-bottom: 8px;"><b>🇮🇳 Gujarati (ગુજરાતી)</b><br><span style="font-size: 12px; opacity: 0.75;">Murmured / breathy vowels with high energy.</span></div>
                    <div style="margin-bottom: 8px;"><b>🇮🇳 Punjabi (ਪੰਜਾਬੀ)</b><br><span style="font-size: 12px; opacity: 0.75;">Tonal Indo-Aryan language with distinct pitch contours.</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="panel-box" style="border-top: 4px solid #34d399;">
                <h3 style="color: #34d399 !important; margin-top: 0;">🏰 Germanic & 🐉 Sino-Tibetan</h3>
                <p style="font-size: 13px; opacity: 0.85;">Major world families serving as global linguistic benchmarks.</p>
                <div style="margin-top: 12px;">
                    <div style="margin-bottom: 8px;"><b>🇬🇧 English (Germanic)</b><br><span style="font-size: 12px; opacity: 0.75;">Stress-timed rhythm, complex diphthongs.</span></div>
                    <div style="margin-bottom: 8px;"><b>🇩🇪 German (Germanic)</b><br><span style="font-size: 12px; opacity: 0.75;">Glottal stops, consonant clusters, distinct umlaut vowel shifts.</span></div>
                    <div style="margin-bottom: 8px;"><b>🇨🇳 Mandarin (Sino-Tibetan)</b><br><span style="font-size: 12px; opacity: 0.75;">Four lexical tonal sweeps with high fundamental frequency variance.</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # PAGE 5: MODEL ENGINE
    elif menu == "⚙️ Model Engine":
        st.markdown("""
        <div class="hero-container">
            <span class="hero-badge">⚙️ Architecture & Engine</span>
            <div class="hero-title">Model Management & Training</div>
            <div class="hero-subtitle">Inspect the deep LSTM network topology, evaluate model performance metrics, or retrain the network.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='section-title'>🧠 Neural Network Topology</div>", unsafe_allow_html=True)
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.markdown("<div class='stat-card'><div style='font-size: 12px; opacity: 0.8;'>Input Layer</div><div style='font-size: 16px; font-weight: 700;'>(1000, 64)</div><div style='font-size: 11px; opacity: 0.6;'>10s @ 100fps</div></div>", unsafe_allow_html=True)
        t2.markdown("<div class='stat-card'><div style='font-size: 12px; opacity: 0.8;'>LSTM Layer 1</div><div style='font-size: 16px; font-weight: 700; color: #0284c7;'>64 Units</div><div style='font-size: 11px; opacity: 0.6;'>Seq-to-Seq</div></div>", unsafe_allow_html=True)
        t3.markdown("<div class='stat-card'><div style='font-size: 12px; opacity: 0.8;'>LSTM Layer 2</div><div style='font-size: 16px; font-weight: 700; color: #6366f1;'>32 Units</div><div style='font-size: 11px; opacity: 0.6;'>Seq-to-Seq</div></div>", unsafe_allow_html=True)
        t4.markdown("<div class='stat-card'><div style='font-size: 12px; opacity: 0.8;'>Dense Hidden</div><div style='font-size: 16px; font-weight: 700; color: #a855f7;'>100 Units</div><div style='font-size: 11px; opacity: 0.6;'>tanh</div></div>", unsafe_allow_html=True)
        t5.markdown(f"<div class='stat-card'><div style='font-size: 12px; opacity: 0.8;'>Softmax Output</div><div style='font-size: 16px; font-weight: 700; color: #10b981;'>{len(LANGUAGES)} Classes</div><div style='font-size: 11px; opacity: 0.6;'>Probabilities</div></div>", unsafe_allow_html=True)
        
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        if st.button(f"🚀 Re-train {len(LANGUAGES)}-Language Model Now", type="primary"):
            with st.spinner("Training LSTM network on speech MFCC features..."):
                import subprocess
                res = subprocess.run(["python", "language_identification.py"], capture_output=True, text=True)
                st.code(res.stdout)
                if res.returncode == 0:
                    st.success("Model retrained successfully!")
                    st.session_state["cached_model"] = None
                    st.rerun()
                else:
                    st.error("Training error:")
                    st.code(res.stderr)

# Entry Point
if __name__ == "__main__":
    is_dark = (st.session_state["theme"] == "🌙 Dark Mode")
    st.markdown(get_theme_css(is_dark), unsafe_allow_html=True)
    
    if not st.session_state["authenticated"]:
        render_login(is_dark)
    else:
        render_dashboard(is_dark)
