import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_model():
    model_name = "sunnypirzada/ag_news_classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()
labels = ["World", "Sports", "Business", "Sci/Tech"]

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
    return labels[pred_label], probs[0][pred_label].item(), probs

st.set_page_config(page_title="News Classifier", page_icon="📰")
st.title("📰 News Topic Classifier (BERT Fine-tuned on AG News)")

user_input = st.text_area("✍️ Paste your news headline or article here:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        label, confidence, probs = predict(user_input)
        st.success(f"**Prediction:** {label} ({confidence*100:.2f}% confidence)")

        st.subheader("Class Probabilities")
        prob_dict = {labels[i]: f"{probs[0][i]*100:.2f}%" for i in range(len(labels))}
        st.json(prob_dict)

