# 📧 Gmail Spam Classifier 

A complete **end-to-end Email Spam Classification System** built using **Machine Learning (Naive Bayes + TF-IDF)** 
and a **Streamlit Web App** that fetches real emails from **Gmail via IMAP** and classifies them as **SPAM or NOT SPAM** 
with adjustable threshold control.

---

## 🚀 Features

    - ✅ Fetches **live emails from Gmail using IMAP**
    - ✅ **Spam / Not Spam classification** using ML
    - ✅ **Naive Bayes + TF-IDF Vectorization**
    - ✅ **Adjustable spam threshold** from the UI
    - ✅ Shows **Spam Probability**
    - ✅ Displays **Model Accuracy, Precision, Recall, F1**
    - ✅ Secure credential handling using **.env**
    - ✅ Clean and professional **Streamlit UI**

---

## 🧠 Machine Learning Model

- **Algorithm:** Multinomial Naive Bayes  
- **Feature Extraction:** TF-IDF Vectorizer  
- **Training Data:** `data/emails.csv`
- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1 Score

The trained model and vectorizer are saved locally and loaded into the Streamlit app for real-time predictions.

---
## 🏗️ Project Structure
'''
Email-spam-classifier/
│
├── app.py # Streamlit Web App
├── imap_gmail.py # Gmail IMAP Email Fetcher
├── train_model.py # ML Model Training Script
├── data/
│ └── emails.csv # Training Dataset
'''

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

    git clone https://github.com/your-username/Email-spam-classifier.git
    
    cd Email-spam-classifier

2️⃣ Install Dependencies

    pip install -r requirements.txt

3️⃣ Enable IMAP in Gmail

    Open Gmail
    Go to Settings → Forwarding & POP/IMAP
    Enable IMAP

4️⃣ Generate Gmail App Password

    Go to Google Account → Security
    Enable 2-Step Verification
    Create App Password → Mail → Windows
    Copy the 16-digit password

5️⃣ Create a file named .env in the project root:
    EMAIL_ACCOUNT=your_email@gmail.com
    EMAIL_PASSWORD=your_16_digit_app_password

🏋️ Train the Machine Learning Model
      python train_model.py


▶️ Run the Web Application

    streamlit run app.py


🎚️ Spam Threshold Control

    The app includes a slider (0.1 – 0.9) to control how strict the spam filter is:
    Lower Threshold (0.3–0.5): More aggressive spam detection
    Higher Threshold (0.7–0.9): Fewer false positives



🛠️ Future Improvements
    
    📊 Promotion vs Spam vs Primary (Multi-class classifier)
    🧑 User feedback based retraining
    ☁️ Cloud deployment (Render / AWS / GCP)
    🔄 Auto refresh inbox
    🗑️ Auto delete spam emails
    🔍 Explainable AI (why an email was marked spam)

👨‍💻 Author

    Gaurav Kumar
    Machine Learning & AI Enthusiast
    GitHub: https://github.com/G-Kumar1
