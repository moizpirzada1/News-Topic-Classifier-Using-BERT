{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "1a001fb4-8e93-48b5-9630-df4447d6bf4d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import torch\n",
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n",
    "import torch.nn.functional as F\n",
    "\n",
    "# -------------------------------\n",
    "# Load Model and Tokenizer\n",
    "# -------------------------------\n",
    "@st.cache_resource\n",
    "def load_model():\n",
    "    model = AutoModelForSequenceClassification.from_pretrained(\"bert-base-uncased\", num_labels=4)\n",
    "    tokenizer = AutoTokenizer.from_pretrained(\"bert-base-uncased\")\n",
    "    return model, tokenizer\n",
    "\n",
    "model, tokenizer = load_model()\n",
    "model.eval()\n",
    "\n",
    "# Define labels (AG News dataset has 4 classes)\n",
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
    "    return labels[pred_label], probs[0][pred_label].item()\n",
    "\n",
    "# -------------------------------\n",
    "# Streamlit App UI\n",
    "# -------------------------------\n",
    "st.set_page_config(page_title=\"News Topic Classifier\", page_icon=\"📰\")\n",
    "\n",
    "st.title(\"📰 News Topic Classifier using BERT\")\n",
    "st.write(\"Enter a news headline or article text, and the model will classify it into one of the categories:\")\n",
    "\n",
    "# Text input\n",
    "user_input = st.text_area(\"✍️ Paste your news headline or article here:\", \"\")\n",
    "\n",
    "if st.button(\"Classify\"):\n",
    "    if user_input.strip() == \"\":\n",
    "        st.warning(\"⚠️ Please enter some text to classify.\")\n",
    "    else:\n",
    "        label, confidence = predict(user_input)\n",
    "        st.success(f\"**Prediction:** {label} ({confidence*100:.2f}% confidence)\")\n",
    "\n",
    "        # Show probabilities for all classes\n",
    "        st.subheader(\"Class Probabilities\")\n",
    "        inputs = tokenizer(user_input, return_tensors=\"pt\", truncation=True, padding=True, max_length=256)\n",
    "        with torch.no_grad():\n",
    "            outputs = model(**inputs)\n",
    "            probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()\n",
    "\n",
    "        prob_dict = {labels[i]: f\"{probs[i]*100:.2f}%\" for i in range(len(labels))}\n",
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
