# Automated ID Card Compliance Monitoring System

![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)
![Status](https://img.shields.io/badge/Status-Fully%20Functional-brightgreen.svg)
![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)

## 🚀 Project Overview

This project implements a real-time, web-based system designed to monitor and enforce institutional ID card compliance. It automatically detects persons, checks for the visible display of an ID card, and employs advanced facial recognition (ArcFace) as a fallback mechanism for unauthorized or non-compliant individuals.

The system features automated logging of violations, configuration-driven model loading, integrated email notifications with proof of violation, and a separate audit trail for all fineable events.

## ✨ Key Features

* **Multi-Stage Detection**: Uses YOLOv8 for efficient detection of Persons and ID Cards.
* **ArcFace Recognition**: Employs InsightFace with ArcFace model for high-accuracy facial recognition when an ID card is not visible.
* **Automated Policy Enforcement**: Issues fines recorded in `students_db.csv` after face identity confirmation.
* **Real-Time Web Interface**: A Flask-based live dashboard displays video feed and system stats.
* **Automated Notifications**: Fine emails sent with attached proof image.
* **Audit Trail**: All finable events logged to `fined_log.csv` with timestamp and image proof.
* **Configurable**: All settings stored in `config.ini`.

## 🛠️ Setup & Installation

### ✅ Prerequisites

* Python 3.9+
* Trained/selected ID Card detection model (`my_model.pt` or similar)
* `students_db.csv` with student details and email addresses
* Face embeddings file: `known_embeddings.npy`

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Configure the System

Ensure correct values in `config.ini`:

* `[SETTINGS]`: camera index
* `[MODELS]`: YOLO model paths
* `[DATABASE]`: DB and embeddings paths
* `[ARCFACE]`: model + threshold
* `[EMAIL]`: SMTP login details
* `[LOGGING]`: fined images and fined log paths

### 3️⃣ Generate Face Embeddings

```bash
python generate_embeddings.py
```

### 4️⃣ Run the Application

```bash
python app.py
```

## 💻 Usage

1. Visit `http://127.0.0.1:5000`
2. Start camera from dashboard
3. Violations automatically logged and emailed
4. Stop camera to free resources

## 📂 Project Structure

```
.
├── app.py
├── config.ini
├── config_loader.py
├── database_manager.py
├── fined_log_manager.py
├── fined_log.csv
├── generate_embeddings.py
├── image_processor.py
├── email_notifier.py
├── utils.py
├── requirements.txt
├── models/
├── captured_images/
├── students_db.csv
├── known_embeddings.npy
└── README.md
```

## ⚠️ Known Issues & Limitations

* Face accuracy depends on embedding quality & threshold tuning
* SMTP providers may limit automated mail volume
* Real-time video depends on system performance
* Captured images require privacy compliance


