import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(99,102,241,0.22), transparent 35%),
        radial-gradient(circle at top right, rgba(236,72,153,0.18), transparent 35%),
        linear-gradient(135deg, #050816 0%, #0B1020 45%, #130B24 100%);
    color: #F8FAFC;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1300px;
}

/* Hide Streamlit style */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Hero Section */
.hero {
    position: relative;
    padding: 45px 45px;
    border-radius: 32px;
    background: linear-gradient(135deg, rgba(255,255,255,0.13), rgba(255,255,255,0.04));
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 25px 70px rgba(0,0,0,0.35);
    overflow: hidden;
    margin-bottom: 28px;
}

.hero::before {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    background: linear-gradient(135deg, #8B5CF6, #EC4899);
    filter: blur(95px);
    opacity: 0.45;
    right: -70px;
    top: -80px;
    animation: floatGlow 5s ease-in-out infinite alternate;
}

@keyframes floatGlow {
    from { transform: translateY(0px) scale(1); }
    to { transform: translateY(35px) scale(1.08); }
}

.hero-badge {
    display: inline-flex;
    gap: 10px;
    align-items: center;
    padding: 9px 16px;
    border-radius: 50px;
    background: rgba(99,102,241,0.18);
    border: 1px solid rgba(165,180,252,0.35);
    color: #C7D2FE;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 54px;
    font-weight: 900;
    line-height: 1.1;
    background: linear-gradient(90deg, #FFFFFF, #C7D2FE, #FBCFE8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    max-width: 760px;
    margin-top: 16px;
    color: #CBD5E1;
    font-size: 18px;
    line-height: 1.7;
}

.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 24px;
}

.pill {
    padding: 10px 16px;
    border-radius: 50px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: #E2E8F0;
    font-size: 14px;
    font-weight: 700;
}

/* Cards */
.glass-card {
    padding: 28px;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.045));
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 20px 55px rgba(0,0,0,0.28);
    min-height: 100%;
}

.section-title {
    font-size: 25px;
    font-weight: 850;
    margin-bottom: 10px;
    color: #FFFFFF;
}

.muted {
    color: #94A3B8;
    font-size: 15px;
    line-height: 1.7;
}

/* Streamlit input styling */
textarea {
    background: rgba(255,255,255,0.94) !important;
    color: #111827 !important;
    border-radius: 18px !important;
    border: 2px solid rgba(139,92,246,0.35) !important;
    font-size: 16px !important;
}

.stTextArea label {
    color: #CBD5E1 !important;
    font-weight: 700 !important;
}

.stButton > button {
    background: linear-gradient(90deg, #7C3AED, #EC4899) !important;
    color: white !important;
    border: none !important;
    border-radius: 18px !important;
    padding: 0.85rem 1rem !important;
    font-weight: 850 !important;
    font-size: 16px !important;
    box-shadow: 0 14px 35px rgba(236,72,153,0.28);
    transition: 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 18px 45px rgba(236,72,153,0.38);
}

/* Result Card */
.result-card {
    margin-top: 26px;
    padding: 34px;
    border-radius: 32px;
    background: linear-gradient(135deg, rgba(34,197,94,0.22), rgba(16,185,129,0.08));
    border: 1px solid rgba(74,222,128,0.30);
    box-shadow: 0 25px 65px rgba(16,185,129,0.18);
    text-align: center;
}

.result-card-negative {
    margin-top: 26px;
    padding: 34px;
    border-radius: 32px;
    background: linear-gradient(135deg, rgba(239,68,68,0.25), rgba(244,63,94,0.08));
    border: 1px solid rgba(248,113,113,0.32);
    box-shadow: 0 25px 65px rgba(239,68,68,0.18);
    text-align: center;
}

.result-label {
    font-size: 38px;
    font-weight: 900;
    color: #FFFFFF;
    margin-bottom: 10px;
}

.result-sub {
    font-size: 17px;
    color: #D1FAE5;
}

.result-sub-negative {
    font-size: 17px;
    color: #FEE2E2;
}

/* Metric Cards */
.metric-card {
    padding: 24px;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(255,255,255,0.13), rgba(255,255,255,0.05));
    border: 1px solid rgba(255,255,255,0.13);
    box-shadow: 0 18px 45px rgba(0,0,0,0.25);
    text-align: center;
    margin-top: 16px;
}

.metric-value {
    font-size: 31px;
    font-weight: 900;
    color: #FFFFFF;
}

.metric-label {
    font-size: 14px;
    color: #94A3B8;
    margin-top: 6px;
}

/* Confidence Bar */
.progress-wrap {
    width: 100%;
    height: 18px;
    background: rgba(255,255,255,0.12);
    border-radius: 999px;
    overflow: hidden;
    margin-top: 18px;
    border: 1px solid rgba(255,255,255,0.12);
}

.progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #22C55E, #A3E635);
    box-shadow: 0 0 25px rgba(34,197,94,0.55);
}

.progress-fill-negative {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #EF4444, #FB7185);
    box-shadow: 0 0 25px rgba(239,68,68,0.55);
}

/* Gauge */
.gauge-box {
    margin-top: 24px;
    padding: 22px;
    border-radius: 24px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
}

.gauge-track {
    width: 100%;
    height: 14px;
    background: linear-gradient(90deg, #EF4444 0%, #F59E0B 50%, #22C55E 100%);
    border-radius: 999px;
    position: relative;
    margin-top: 16px;
}

.gauge-marker {
    position: absolute;
    top: -8px;
    width: 4px;
    height: 30px;
    background: #FFFFFF;
    border-radius: 999px;
    box-shadow: 0 0 18px rgba(255,255,255,0.8);
}

/* Sample Cards */
.sample-title {
    font-size: 24px;
    font-weight: 850;
    color: #FFFFFF;
    margin: 34px 0 14px;
}

.sample-card {
    padding: 20px;
    border-radius: 22px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.13);
    color: #E2E8F0;
    min-height: 120px;
    line-height: 1.6;
}

.footer {
    margin-top: 35px;
    padding: 24px;
    border-radius: 24px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    text-align: center;
    color: #94A3B8;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL AND WORD INDEX
# =====================================================
@st.cache_resource
def load_sentiment_model():
    return load_model("sentiment_rnn_model.keras")

@st.cache_data
def load_word_index():
    with open("word_index.pkl", "rb") as file:
        return pickle.load(file)

model = load_sentiment_model()
word_index = load_word_index()

max_length = 500
vocab_size = 10000

# =====================================================
# TEXT PREPROCESSING
# =====================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

def encode_review(text):
    text = clean_text(text)
    words = text.split()

    encoded_review = []
    for word in words:
        index = word_index.get(word, 2)

        if index < vocab_size:
            encoded_review.append(index + 3)
        else:
            encoded_review.append(2)

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=max_length,
        padding="pre",
        truncating="pre"
    )

    return padded_review

def predict_sentiment(review):
    encoded_review = encode_review(review)
    probability = float(model.predict(encoded_review, verbose=0)[0][0])

    if probability >= 0.5:
        sentiment = "Positive"
        confidence = probability * 100
    else:
        sentiment = "Negative"
        confidence = (1 - probability) * 100

    return sentiment, confidence, probability

# =====================================================
# SESSION STATE FOR CLICKABLE SAMPLES
# =====================================================
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

def set_sample_review(text):
    st.session_state.review_text = text

positive_sample = "The movie was absolutely wonderful. The story was emotional, the acting was brilliant, and the ending was deeply satisfying."
negative_sample = "The movie was disappointing and slow. The story felt weak, the acting was dull, and I completely lost interest."

# =====================================================
# HERO SECTION
# =====================================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">● AI Movie Review Classifier</div>
    <div class="hero-title">Sentiment Intelligence<br>Powered by SimpleRNN</div>
    <div class="hero-subtitle">
        Analyze IMDb-style movie reviews using a Recurrent Neural Network that learns sequential word patterns
        and predicts whether the sentiment is positive or negative.
    </div>
    <div class="pill-row">
        <div class="pill">SimpleRNN</div>
        <div class="pill">Deep Learning</div>
        <div class="pill">NLP</div>
        <div class="pill">IMDb Dataset</div>
        <div class="pill">Binary Classification</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# MAIN DASHBOARD
# =====================================================
left, right = st.columns([1.45, 1])

with left:
    st.markdown("""
    <div class="glass-card">
        <div class="section-title">Write a Movie Review</div>
        <div class="muted">
            Enter a review below and let the RNN model identify the emotional tone behind the text.
        </div>
    </div>
    """, unsafe_allow_html=True)

    user_review = st.text_area(
        "Movie review text",
        value=st.session_state.review_text,
        height=230,
        placeholder="Example: The movie was emotional, beautifully directed, and the acting was excellent..."
    )

    analyze = st.button("Analyze Sentiment", use_container_width=True)

with right:
    st.markdown("""
    <div class="glass-card">
        <div class="section-title">Model Overview</div>
        <div class="muted">
            This dashboard uses a trained SimpleRNN model to classify movie reviews into positive or negative sentiment.
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-value">SimpleRNN</div>
            <div class="metric-label">Model Architecture</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">10,000</div>
            <div class="metric-label">Vocabulary Size</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">500</div>
            <div class="metric-label">Sequence Length</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# RESULT SECTION
# =====================================================
if analyze:
    if user_review.strip() == "":
        st.warning("Please enter a movie review before analyzing.")
    else:
        sentiment, confidence, probability = predict_sentiment(user_review)

        if sentiment == "Positive":
            result_class = "result-card"
            result_sub_class = "result-sub"
            progress_class = "progress-fill"
            emoji = "Positive Review"
            tone = "The review expresses a favorable opinion."
            gauge_position = min(max(probability * 100, 0), 100)
        else:
            result_class = "result-card-negative"
            result_sub_class = "result-sub-negative"
            progress_class = "progress-fill-negative"
            emoji = "Negative Review"
            tone = "The review expresses an unfavorable opinion."
            gauge_position = min(max(probability * 100, 0), 100)

        st.markdown(f"""
        <div class="{result_class}">
            <div class="result-label">{emoji}</div>
            <div class="{result_sub_class}">{tone}</div>
            <div class="progress-wrap">
                <div class="{progress_class}" style="width:{confidence:.2f}%;"></div>
            </div>
            <div style="margin-top:14px; color:#E5E7EB; font-weight:800;">
                Confidence Score: {confidence:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{sentiment}</div>
                <div class="metric-label">Predicted Sentiment</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{confidence:.2f}%</div>
                <div class="metric-label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{probability:.4f}</div>
                <div class="metric-label">Positive Probability</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="gauge-box">
            <div class="section-title">Sentiment Gauge</div>
            <div class="muted">
                Left side indicates negative sentiment. Right side indicates positive sentiment.
            </div>
            <div class="gauge-track">
                <div class="gauge-marker" style="left:{gauge_position:.2f}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# CLICKABLE SAMPLE REVIEWS
# =====================================================
st.markdown('<div class="sample-title">Try Clickable Sample Reviews</div>', unsafe_allow_html=True)

s1, s2 = st.columns(2)

with s1:
    st.markdown("""
    <div class="sample-card">
        <b>Positive Sample</b><br><br>
        The movie was absolutely wonderful. The story was emotional, the acting was brilliant,
        and the ending was deeply satisfying.
    </div>
    """, unsafe_allow_html=True)
    st.button(
        "Use Positive Sample",
        use_container_width=True,
        on_click=set_sample_review,
        args=(positive_sample,)
    )

with s2:
    st.markdown("""
    <div class="sample-card">
        <b>Negative Sample</b><br><br>
        The movie was disappointing and slow. The story felt weak, the acting was dull,
        and I completely lost interest.
    </div>
    """, unsafe_allow_html=True)
    st.button(
        "Use Negative Sample",
        use_container_width=True,
        on_click=set_sample_review,
        args=(negative_sample,)
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    Built with Python, TensorFlow/Keras, SimpleRNN and Streamlit.
</div>
""", unsafe_allow_html=True)