# 🎭 Text Emotion & Toxicity Detection using Machine Learning & NLP

🚀 An end-to-end **Natural Language Processing (NLP)** project that detects both **human emotions** and **toxic behavior** from text using Machine Learning.

This system analyzes user input and provides:

### 🎯 Emotion Classification:

😊 Happy | 😢 Sad | 😠 Angry | 😨 Fear | 😲 Surprise | 😐 Neutral

### ⚠️ Toxicity Detection:

🚫 Toxic | ✅ Non-Toxic

---

## 🧠 Project Overview

Understanding both **emotion** and **toxicity** in text is crucial for modern AI systems.

This project combines:

* 🎭 Emotion Detection (multi-class classification)
* ⚠️ Toxic Comment Detection (binary classification)

📌 Real-world applications:

* 💬 Chatbots moderation
* 📱 Social media content filtering
* 📊 Customer feedback analysis
* 🧘 Mental health monitoring
* 🛡️ Online harassment detection

---

## ⚙️ Tech Stack

* 🐍 Python
* 📚 Scikit-learn
* 🧹 NLTK
* 📊 Pandas, NumPy
* 🌐 Streamlit

---

## 🔄 Workflow

1. **Data Collection**

   * Emotion dataset
   * Toxic comment dataset

2. **Text Preprocessing**

   * Tokenization
   * Stopword Removal
   * Stemming

3. **Feature Engineering**

   * TF-IDF Vectorization

4. **Model Building**

   * Emotion Classification Model
   * Toxicity Detection Model

5. **Evaluation**

   * Accuracy
   * Confusion Matrix

6. **Deployment**

   * Streamlit app for real-time predictions

---

## 🚀 Features

✔ Emotion Detection (multi-class)
✔ Toxicity Detection (binary classification)
✔ Clean text preprocessing pipeline
✔ TF-IDF feature extraction
✔ High model performance
✔ Real-time prediction via Streamlit UI

---

## 📂 Project Structure

```id="o6x0b1"
Text-Emotion-Detection/
│
├── data/
│   ├── emotion_dataset.csv
│   └── toxic_dataset.csv
│
├── notebooks/
│   └── emotion_toxic_detection.ipynb
│
├── models/
│   ├── emotion_model.pkl
│   └── toxic_model.pkl
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

### 🔹 Step 1: Clone the repository

```bash id="lqn3rj"
git clone https://github.com/dipakpatil8832/Text-Emotion-detection.git
cd Text-Emotion-detection
```

### 🔹 Step 2: Create virtual environment

```bash id="9ynlcv"
python -m venv .venv
```

### 🔹 Step 3: Activate environment

```bash id="6ohpje"
.venv\Scripts\activate   # Windows
```

### 🔹 Step 4: Install dependencies

```bash id="f3nxte"
pip install -r requirements.txt
```

### 🔹 Step 5: Run Streamlit app

```bash id="6u7dqq"
streamlit run app/app.py
```

---

## 📊 Sample Output

**Input:**
`"I hate this product, it's terrible!"`

**Output:**
👉 Emotion: Angry 😠
👉 Toxicity: Toxic 🚫

---

## 🔮 Future Improvements

* 🔥 Deep Learning (LSTM / BERT)
* 🌍 Multi-language support
* 📊 Advanced dashboard
* ☁️ Cloud deployment (Render / AWS)

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork and improve this project.

---

## 📜 License

MIT License

---

## 👨‍💻 Author

**Dipak Patil**
📌 Data Science & AI Enthusiast

---
