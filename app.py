import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import torch
import joblib
import numpy as np

# Load temperature scalar
T = joblib.load("temperature_scalar.pkl")

# Softmax function for logits
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Country flags (Unicode emojis)
flags = {
    "Argentina": "🇦🇷",
    "Chile": "🇨🇱",
    "Colombia": "🇨🇴",
    "Mexico": "🇲🇽",
    "Spain": "🇪🇸"
}

# Index to country label mapping
dic = {
    0: "Spain",
    1: "Argentina",
    2: "Colombia",
    3: "Mexico",
    4: "Chile"
}

# ✅ Cached model and tokenizer loading
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained(
        "/Users/samuelmarzano/Documentos/Ciencia_de_Datos/Tercero/PROY III/modelo_sintok_V1/modelo_finetuned/",
        local_files_only=True,
        low_cpu_mem_usage=False
    )
    model.eval()
    return model

@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

model = load_model()
tokenizer = load_tokenizer()

# Prediction functions
def predict_label(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

def predict_proba(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, smax_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.cpu().numpy().flatten()

    scaled_logits = logits / T
    probabilities = softmax(scaled_logits)

    return probabilities


# Callback function to clear input
def clear_text():
    st.session_state["text_input"] = ""

# UI Layout
st.markdown("""
    <h1 style="text-align: center; color: #4CAF50;">🌍 LexOrigin</h1>
    <p style="text-align: center; font-size: 16px;">🔍 Identify the origin of text from 5 countries</p>
    <hr style="border:1px solid #ddd;">
""", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; font-size: 14px; color: grey;'>Currently, we classify the following 5 countries:</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>🇦🇷 Argentina | 🇨🇱 Chile | 🇨🇴 Colombia | 🇲🇽 Mexico | 🇪🇸 Spain</p>", unsafe_allow_html=True)

# Input field
text = st.text_area("Enter text for classification:", height=150, key="text_input")

# Buttons
col1, col2 = st.columns([1, 1])
with col1:
    predict_button = st.button("🔮 Predict", use_container_width=True)
with col2:
    clear_button = st.button("❌ Clear", on_click=clear_text, use_container_width=True)

# Prediction logic
if predict_button and text.strip():
    prediction = dic[predict_label(text)]
    confidence_scores = predict_proba(text)
    confidence_map = {
        'Spain': confidence_scores[0],
        'Argentina': confidence_scores[1],
        'Colombia': confidence_scores[2],
        'Mexico': confidence_scores[3],
        'Chile': confidence_scores[4]
    }
    sorted_scores = sorted(confidence_map.items(), key=lambda x: x[1], reverse=True)

    st.success(f"🏆 Prediction: {flags.get(prediction, '🏳')} {prediction}")
    st.info("### 🔢 Confidence Scores:")
    for label, confidence in sorted_scores:
        st.markdown(f"<p style='font-size: 16px;'> {flags.get(label, '🏳')} {label}: {confidence * 100:.2f}%</p>", unsafe_allow_html=True)
