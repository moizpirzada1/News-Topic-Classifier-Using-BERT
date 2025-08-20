import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------
# Load Model and Tokenizer (from HF Hub)
# -------------------------------
@st.cache_resource
def load_model():
    model_name = "sunnypirzada/ag_news_classifier"   # your HF repo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Labels
labels = ["🌍 World", "🏅 Sports", "💼 Business", "🖥️ Sci/Tech"]

# -------------------------------
# Prediction Function
# -------------------------------
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
    return labels[pred_label], probs[0][pred_label].item(), probs

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="News Classifier", page_icon="📰", layout="wide")

# Sidebar info
st.sidebar.title("📰 News Classifier")
st.sidebar.info("This app uses a **BERT model fine-tuned on AG News** to classify headlines or short articles into 4 categories.")

st.markdown("<h1 style='text-align: center;'>📰 News Topic Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Powered by <b>BERT</b> • Fine-tuned on AG News</p>", unsafe_allow_html=True)
st.write("---")

# Initialize history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Input area
user_input = st.text_area("✍️ Paste your news headline or article here:", height=120)

# Center button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    classify_btn = st.button("🚀 Classify", use_container_width=True)

if classify_btn:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        label, confidence, probs = predict(user_input)

        # Show main prediction in a styled box
        st.markdown(
            f"<div style='padding:15px; background-color:#f0f2f6; border-radius:10px; text-align:center;'>"
            f"<h2>🔮 Prediction: <span style='color:#0068c9'>{label}</span></h2>"
            f"<p>Confidence: <b>{confidence*100:.2f}%</b></p>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.write("---")

        # Save to history
        st.session_state.history.append({
            "text": user_input,
            "label": label,
            "confidence": f"{confidence*100:.2f}%",
            "probs": {labels[i]: f"{probs[0][i]*100:.2f}%" for i in range(len(labels))}
        })

        # Show class probabilities
        st.subheader("📊 Class Probabilities")
        cols = st.columns(4)
        for i, lab in enumerate(labels):
            with cols[i]:
                st.metric(label=lab, value=f"{probs[0][i]*100:.2f}%")

# -------------------------------
# Show History
# -------------------------------
if st.session_state.history:
    st.write("---")
    st.subheader("🗂️ Classification History")

    for idx, record in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"{idx}. {record['text'][:50]}..."):
            st.write(f"**Prediction:** {record['label']} ({record['confidence']})")
            st.json(record["probs"])
