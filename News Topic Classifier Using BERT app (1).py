{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7e58d3bf-b957-45d0-ab25-24c6611f09f3",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-08-20 05:38:07.915 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\R Y Z E N\\shopgenie\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import torch\n",
    "import torch.nn.functional as F\n",
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n",
    "\n",
    "# -------------------------------\n",
    "# Load Model and Tokenizer (from HF Hub)\n",
    "# -------------------------------\n",
    "@st.cache_resource\n",
    "def load_model():\n",
    "    model_name = \"sunnypirzada/ag_news_classifier\"   # your HF repo\n",
    "    tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
    "    model = AutoModelForSequenceClassification.from_pretrained(model_name)\n",
    "    model.eval()\n",
    "    return model, tokenizer\n",
    "\n",
    "model, tokenizer = load_model()\n",
    "\n",
    "# Labels\n",
    "labels = [\"World\", \"Sports\", \"Business\", \"Sci/Tech\"]\n",
    "\n",
    "# -------------------------------\n",
    "# Prediction Function\n",
    "# -------------------------------\n",
    "def predict(text):\n",
    "    inputs = tokenizer(text, return_tensors=\"pt\", truncation=True, padding=True, max_length=256)\n",
    "    with torch.no_grad():\n",
    "        outputs = model(**inputs)\n",
    "        probs = F.softmax(outputs.logits, dim=1)\n",
    "        pred_label = torch.argmax(probs, dim=1).item()\n",
    "    return labels[pred_label], probs[0][pred_label].item(), probs\n",
    "\n",
    "# -------------------------------\n",
    "# Streamlit UI\n",
    "# -------------------------------\n",
    "st.set_page_config(page_title=\"News Classifier\", page_icon=\"📰\")\n",
    "st.title(\"📰 News Topic Classifier (BERT Fine-tuned on AG News)\")\n",
    "\n",
    "user_input = st.text_area(\"✍️ Paste your news headline or article here:\")\n",
    "\n",
    "if st.button(\"Classify\"):\n",
    "    if user_input.strip() == \"\":\n",
    "        st.warning(\"⚠️ Please enter some text to classify.\")\n",
    "    else:\n",
    "        label, confidence, probs = predict(user_input)\n",
    "        st.success(f\"**Prediction:** {label} ({confidence*100:.2f}% confidence)\")\n",
    "\n",
    "        # Show probabilities\n",
    "        st.subheader(\"Class Probabilities\")\n",
    "        prob_dict = {labels[i]: f\"{probs[0][i]*100:.2f}%\" for i in range(len(labels))}\n",
    "        st.json(prob_dict)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (shopgenie)",
   "language": "python",
   "name": "shopgenie"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
