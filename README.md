# 📩 Spam Email Classifier

Detect Spam Emails using Machine Learning and NLP — built with Python, Scikit-learn, and Streamlit.

---

## 📌 Project Overview
This project is an end-to-end AI/ML application that classifies an email/SMS message as **Spam** or **Ham (Not Spam)** using a **Multinomial Naive Bayes** classifier trained on the **SMS Spam Collection Dataset**. The model is served through an interactive **Streamlit** web app.

---

## ✨ Features
- Clean, professional and responsive UI
- Text area input for any email / SMS message
- **Predict** and **Clear** buttons
- Loading spinner during prediction
- ✅ Success message for safe (Ham) emails
- 🚨 Warning message for Spam emails
- Confidence percentage with progress bar
- Sidebar with model details, accuracy and tech stack
- Footer section

---

## 🛠️ Technologies Used
- **Python 3.9+**
- **Scikit-learn** — ML model & TF-IDF
- **NLTK** — Text preprocessing / stopwords
- **Pandas / NumPy** — Data handling
- **Streamlit** — Web interface

---

## 📊 Dataset Information
- **Name:** SMS Spam Collection Dataset
- **Source:** UCI Machine Learning Repository
- **Samples:** ~5,574 SMS messages labeled as `ham` or `spam`
- **URL:** https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

The training script auto-downloads the dataset if `spam.csv` is not present.

---

## ⚙️ Installation Steps

```bash
# 1. Clone or extract the project
cd Spam_Email_Classifier

# 2. (Optional) create a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (creates model.pkl, vectorizer.pkl, accuracy.txt)
python train_model.py

# 5. Run the Streamlit app
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

---

## 🚀 Streamlit Cloud Deployment Steps
1. Push the project to a **public GitHub repository**.
2. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**.
4. Select your repository, branch (`main`), and set the main file as **`app.py`**.
5. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically.
6. Your app will be live at: `https://<your-app-name>.streamlit.app`

> ✅ The repo already contains `model.pkl` and `vectorizer.pkl`, so the app works on Streamlit Cloud without re-training.

---

## 📤 GitHub Upload Steps

```bash
cd Spam_Email_Classifier
git init
git add .
git commit -m "Initial commit - Spam Email Classifier"
git branch -M main
git remote add origin https://github.com/<your-username>/Spam_Email_Classifier.git
git push -u origin main
```

---

## 📸 Screenshots
_Add your screenshots here after running the app:_
- `screenshots/home.png` — Home page
- `screenshots/spam_result.png` — Spam detection result
- `screenshots/ham_result.png` — Safe email result

---

## 🔮 Future Improvements
- Support multiple languages
- Add deep-learning models (LSTM / BERT)
- Email file upload (`.eml`, `.txt`) support
- Batch prediction via CSV upload
- User authentication and history tracking
- REST API endpoint for integrations
- Dark mode toggle

---

## 📂 Project Structure

```
Spam_Email_Classifier/
│
├── app.py              # Streamlit web app
├── train_model.py      # Model training pipeline
├── model.pkl           # Trained Naive Bayes model
├── vectorizer.pkl      # TF-IDF vectorizer
├── accuracy.txt        # Saved test accuracy
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

---

Built with ❤️ for an AI/ML Internship Project.
