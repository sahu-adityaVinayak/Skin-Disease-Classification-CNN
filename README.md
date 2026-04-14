# Skin Disease Classification using Deep Learning and CNN 🔬🤖

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green.svg)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-CSS-38B2AC.svg)

## 📌 Project Overview
This project leverages Deep Learning and Convolutional Neural Networks (CNNs) to accurately classify skin lesions. By utilizing a highly optimized **MobileNet** architecture, the model acts as an automated screening tool to differentiate between **Malignant** (Dangerous/Cancerous) and **Benign** (Safe/Harmless) skin conditions with a **90% validation accuracy**.

Developed by a dedicated team of four, this project features a robust machine learning backend seamlessly integrated with a modern, responsive web application (Flask + Tailwind CSS) for real-time medical image analysis.

## Model Architecture & Strategy
* **Base Model:** MobileNet (Pre-trained on ImageNet)
* **Dataset:** HAM10000 (10,015 dermoscopic images)
* **Master Strategy (Binary Classification):** To overcome the notorious extreme class imbalance of the HAM10000 dataset, the original 7 categories were mapped and balanced into 2 critical real-world classes:
  * `Benign`: Actinic keratoses, Basal cell carcinoma, Benign keratosis-like lesions, Dermatofibroma
  * `Malignant`: Melanoma, Melanocytic nevi, Vascular lesions
* **Data Augmentation:** Applied random oversampling and extensive image augmentation (rotation, zooming, flipping) to prevent catastrophic forgetting and overfitting.
* **Hardware:** Trained efficiently using Kaggle's Dual GPU T4x2 environment.

## Key Features
* **High Accuracy:** Consistently achieves **~90% accuracy** on validation datasets, outperforming several baseline research papers on the same dataset.
* **Interactive Frontend:** A sleek, glass-morphism UI built with Tailwind CSS that allows users to upload dermoscopic images and instantly view AI predictions.
* **Instant Confidence Scoring:** The backend calculates and displays the exact probability/confidence percentage of the diagnosis.
* **Lightweight & Fast:** The use of MobileNet ensures the `.h5` model file is extremely compact, allowing for rapid predictions without heavy local GPU requirements.

## Project Structure
DermScan-AI/
│
├── model/
│   └── model.h5                 # The trained MobileNet binary model
├── static/
│   └── uploads/                 # Temporary folder for user-uploaded images
├── templates/
│   └── index.html               # Frontend UI (Tailwind CSS)
├── app.py                       # Flask Backend handling routing and model inference
└── README.md                    # Project documentation

1. Clone the repository:

git clone [https://github.com/sahu-adityaVinayak/Skin-Disease-Classification-CNN.git](https://github.com/sahu-adityaVinayak/Skin-Disease-Classification-CNN.git)
cd Skin-Disease-Classification-CNN

2. Install required dependencies:

pip install flask tensorflow numpy werkzeug pillow

3. Run the Flask Server:

python app.py

4. Open in Browser:

Navigate to http://127.0.0.1:5000/ in your web browser. Upload a dermoscopic image and click "Analyze with AI".

👥 Acknowledgements

This project was conceptualized and developed as a collaborative University Research Project. Special thanks to our guides and the open-source Kaggle community for dataset access.
