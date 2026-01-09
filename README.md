# Mango_Disease_Detection

Mango Leaf & Fruit Disease Detection using DRIAF Hybrid Deep Learning Model

A Deep Learning–powered system for early detection of mango crop diseases.

📌 Project Overview

This project aims to detect mango leaf and fruit diseases using a custom hybrid deep learning model (DRIAF) that combines:

DenseNet121 , ResNet50 , InceptionV3 were 
These three feature extractors are fused (concatenated) to form the DRIAF Hybrid Model, enabling high-accuracy multi-disease classification.

Farmers or researchers can upload an image of a mango leaf or fruit, and the system will predict:

✔ Disease name
✔ Confidence score
✔ Description of the disease
✔ Recommended preventive solutions


This helps in fast diagnosis, reducing crop loss, and supporting smart agriculture.

 
 Project Goals

Automate mango disease detection using AI

Achieve high accuracy using hybrid feature extraction

Build a simple UI for uploading plant images

Provide actionable disease descriptions & treatments

Make the system deployable on local machines or cloud

 
DRIAF Hybrid Model Architecture

The model combines three pre-trained CNNs:

DenseNet121  → ┐
ResNet50     → ├── Concatenate → Fully Connected Layers → Softmax
InceptionV3  → ┘


This architecture extracts richer features → increasing accuracy & robustness.

📂 Dataset Used
✔ 1. Mango Leaf Disease Dataset

Source: Kaggle
Contains the following classes:

Anthracnose

Bacterial Canker

Cutting Weevil

Die Back

Gall Midge

Healthy

Powdery Mildew

Sooty Mould

✔ 2. Mango Fruit Dataset

Source: Kaggle
Contains multiple mango fruit diseases + healthy samples.




⚙ Model Training Workflow
STEP 1 — Select & Download Dataset

Downloaded directly using kagglehub (no manual upload required)

Used Google Colab for GPU acceleration

STEP 2 — Preprocessing

Image resizing (224 × 224)

Normalization (1/255)

Train/Validation/Test split (80–20)

STEP 3 — Build DRIAF Model

Load DenseNet121, ResNet50, InceptionV3 without top layers

Freeze initial layers

Merge features

Add Dense layers, Dropout, Softmax

STEP 4 — Training

Optimizer: Adam

Loss: Categorical Crossentropy

Metrics: Accuracy

STEP 5 — Evaluation

Confusion matrix

Classification report

Accuracy / loss graphs

Best model saved as driaf_baseline.h5

STEP 6 — Deployment



Backend:

TensorFlow

Flask / Streamlit

Backend API served your trained model

Frontend:

HTML

CSS

JavaScript

Upload image → backend predicts → returns result

🖥 Tech Stack
Component	Technology
Model	TensorFlow / Keras
Training	Google Colab GPU
Dataset	Kaggle
Backend	Flask / Streamlit
Frontend	HTML + CSS + JS
Deployment	Local Machine / GitHub
📷 User Interface (Streamlit Version)

Upload any mango leaf/fruit image

Real-time prediction

Confidence score

Disease details

Remedies and solutions provided



 Supported Diseases
Disease	Description
Anthracnose	Fungal infection causing dark spots
Powdery Mildew	White powder appearance on leaf surface
Gall Midge	Insect-induced leaf tumors
Die Back	Gradual drying of twigs
Bacterial Canker	Water-soaked lesions and cracking
Cutting Weevil	Insect cutting edges of leaves
Sooty Mould	Black fungal coating
Healthy	No disease detected



Each disease has prevention & treatment steps in the UI.

🚀 Features

✔ Detects 8 different mango plant diseases
✔ Works for leaf & fruit images
✔ High accuracy using hybrid CNN
✔ Runs locally (offline)
✔ Frontend + Backend integrated
✔ Real-time prediction
✔ Disease solutions included

📌 Project Structure
Mango_Disease_Detection/
│── backend/
│   ├── app.py
│   ├── model/
│   │   └── DRIAF_models/
│   │       └── driaf_baseline.h5
│── frontend/
│   ├── index.html
│   ├── script.js
│   ├── style.css
│── .venv/
│── requirements.txt

🏁 How to Run Locally
1. Activate Virtual Environment
.\.venv\Scripts\activate

2. Go to Backend Folder
cd backend

3. Start Application
streamlit run app.py




Results (Model Performance)

Accuracy: (insert your score here)

Loss: (insert value)








Confusion Matrix: (generated in notebook)

DRIAF outperformed individual models (DenseNet, ResNet, Inception)



Our Website Link : https://mangoplantdisease.streamlit.app/

(When you provide your real metrics, I will fill this section.)
