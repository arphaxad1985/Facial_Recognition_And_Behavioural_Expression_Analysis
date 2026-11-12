# 🎭 ProctorAI: Real-Time Behavioral Analysis System

## 🚀 Overview

This project implements a real-time facial recognition and behavioral analysis system. It is designed to identifying emotions and detecting potential suspicious activities (malpractice) in examination settings.

The system uses:
- **Facial Expression Recognition (FER)**: To detect emotions like angry, happy, sad, neutral, etc.
- **Object Detection (YOLOv8)**: To detect unauthorized objects like mobile phones.

## 🛠️ Features

✔ **Live Dashboard**: A modern web-based interface for monitoring.
✔ **Expression Analysis**: Real-time classification of user emotions.
✔ **Malpractice Detection**: Automatically flags "Cell Phone" usage with red bounding boxes.
✔ **Session Stats**: Tracks session time and activity.

## 📦 Tech Stack

- **Backend**: Python, Flask, OpenCV
- **AI/ML**: FER (Facial Expression Recognition), Ultralytics YOLOv8
- **Frontend**: HTML5, CSS3 (Modern Dark UI)

## 🏃‍♂️ How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Application**:
   ```bash
   python app.py
   ```
   Or simply double-click `run.bat`.

3. **Open Browser**:
   Navigate to `http://localhost:5000` to view the dashboard.

## 📝 Research & Citations

> I have done a complete research about on AI Pioneering Ethical, Analytical and Real time Emotional Recognition in Dynamic Human Expressions, which has been published in IEEE in the journal of ICCCDSAI 2025 (DOI: 10.1109/ICDSAAI65575.2025.11011864). 

Attribution

Starter pipeline structure adapted from an open-source facial 
expression recognition project. Model training (28,000+ images), 
detection pipeline, and deployment implemented by Arphaxad Nguka Owange.

