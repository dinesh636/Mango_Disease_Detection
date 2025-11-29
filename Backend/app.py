""" 

import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image

# =====================================================
# ✅ 1. Initialize Flask app
# =====================================================
app = Flask(__name__)

# ✅ Model path (adjusted for your folder)
MODEL_PATH = os.path.join(os.getcwd(), "model", "DRIAF_models", "driaf_baseline.h5")

print(f"🔍 Checking model path: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}")
else:
    print(f"✅ Model found at: {MODEL_PATH}")

# ✅ Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ DRIAF Model Loaded Successfully!")

# Class names (update if your model has different ones)
CLASS_NAMES = ["Healthy", "Anthracnose", "Bacterial Canker", "Powdery Mildew", "Sooty Mould"]

# =====================================================
# ✅ 2. Prediction API
# =====================================================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    return jsonify({
        'class': predicted_class,
        'confidence': f"{confidence:.2f}%"
    })

# =====================================================
# ✅ 3. Run Flask app
# =====================================================
if __name__ == '__main__':
    print("✅ Flask Mango Disease Detection Backend is Running!")
    app.run(debug=True)


    
    



import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# =====================================================
# 🧩 1. App Configuration
# =====================================================
st.set_page_config(
    page_title="Mango Disease Detection - DRIAF",
    page_icon="🥭",
    layout="centered"
)

st.title("🍃 Mango Leaf & Fruit Disease Detection using DRIAF Hybrid Model")
st.write("Upload a mango leaf or fruit image to detect possible diseases.")

# =====================================================
# 📁 2. Load DRIAF Model Safely (Handles Windows Paths)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "DRIAF_models", "driaf_baseline.h5")

st.write(f"🔍 Looking for model at: `{MODEL_PATH}`")

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ DRIAF model loaded successfully!")
else:
    st.error(f"❌ Model not found! Expected at: {MODEL_PATH}")
    st.stop()



#demo
# --------------------------------------------------------------------
# 🏷 DISEASE LABELS + SOLUTIONS
# --------------------------------------------------------------------
DISEASE_INFO = {
    "Anthracnose": {
        "cause": "A fungal infection caused by Colletotrichum species.",
        "solution": "Apply copper-based fungicides, improve ventilation, remove infected leaves."
    },
    "Bacterial Canker": {
        "cause": "Caused by Xanthomonas bacteria, spreads during humid weather.",
        "solution": "Use copper oxychloride spray, prune infected parts, avoid overhead irrigation."
    },
    "Cutting Weevil": {
        "cause": "Caused by insect larvae that feed on leaf tissues.",
        "solution": "Use neem oil spray, pheromone traps, and biological pest control."
    },
    "Die Back": {
        "cause": "Fungal infection starting from leaf tip and spreading backward.",
        "solution": "Apply Bordeaux mixture, prune dried branches, improve soil drainage."
    },
    "Gall Midge": {
        "cause": "Tiny insects laying eggs on tender leaves, forming galls.",
        "solution": "Use light traps, systemic insecticides, remove affected leaves."
    },
    "Healthy": {
        "cause": "No disease detected. The leaf appears healthy.",
        "solution": "Maintain regular watering, fertilization, and pest monitoring."
    },
    "Powdery Mildew": {
        "cause": "Fungal spores causing white powdery coating on leaves.",
        "solution": "Spray sulfur fungicides, avoid high humidity, prune crowded branches."
    },
    "Sooty Mould": {
        "cause": "Black fungus growing on honeydew secreted by pests like aphids.",
        "solution": "Control pests using neem oil; wash leaves with mild soap solution."
    }
}

CLASS_NAMES = list(DISEASE_INFO.keys())





#demo


# =====================================================
# 🧠 3. Define Class Labels
# =====================================================
CLASS_NAMES = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould",
    "Healthy"
]

# =====================================================
# 🖼️ 4. Image Upload & Preprocessing
# =====================================================
uploaded_file = st.file_uploader("📸 Upload an image (Leaf or Fruit)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # =====================================================


    # 🔮 5. Make Prediction
    # =====================================================
    try:
        prediction = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.subheader("🔍 Prediction Result")
        st.success(f"**{predicted_class}**")
        # demo
        st.write(f"### 🔍 Why This Happens\n{DISEASE_INFO[predicted_class]['cause']}")
        st.write(f"### 🌱 Solution\n{DISEASE_INFO[predicted_class]['solution']}")   

        # st.success(f"**{predicted_class}** ({confidence:.2f}% confidence)")

                
# Show cause + solution
           # st.markdown("<div class='solution-box'>", unsafe_allow_html=True)
           # st.write(f"### 🔍 Why This Happens\n{DISEASE_INFO[predicted_class]['cause']}")
           #  st.write(f"### 🌱 Solution\n{DISEASE_INFO[predicted_class]['solution']}")
             #  st.markdown("</div>", unsafe_allow_html=True)

            # st.markdown("</div>", unsafe_allow_html=True)




    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

"""



import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# =====================================================
# 🧩 1. App Configuration
# =====================================================
st.set_page_config(
    page_title="Mango Disease Detection - DRIAF",
    page_icon="🥭",
    layout="centered"
)

st.title("🍃 Mango Leaf & Fruit Disease Detection using DRIAF Hybrid Model")
st.write("Upload a mango leaf or fruit image to detect possible diseases.")

# =====================================================
# 📁 2. Load DRIAF Model Safely
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "DRIAF_models", "driaf_baseline.h5")

st.write(f"🔍 Looking for model at: `{MODEL_PATH}`")

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ DRIAF model loaded successfully!")
else:
    st.error(f"❌ Model not found! Expected at: {MODEL_PATH}")
    st.stop()

# =====================================================
# 🏷 3. DISEASE LABELS + SOLUTIONS
# =====================================================
DISEASE_INFO = {
    "Anthracnose": {
        "cause": "A fungal infection caused by Colletotrichum species.",
        "solution": "Apply copper-based fungicides, improve ventilation, remove infected leaves."
    },
    "Bacterial Canker": {
        "cause": "Caused by Xanthomonas bacteria, spreads during humid weather.",
        "solution": "Use copper oxychloride spray, prune infected parts, avoid overhead irrigation."
    },
    "Cutting Weevil": {
        "cause": "Caused by insect larvae that feed on leaf tissues.",
        "solution": "Use neem oil spray, pheromone traps, and biological pest control."
    },
    "Die Back": {
        "cause": "Fungal infection starting from leaf tip and spreading backward.",
        "solution": "Apply Bordeaux mixture, prune dried branches, improve soil drainage."
    },
    "Gall Midge": {
        "cause": "Tiny insects laying eggs on tender leaves, forming galls.",
        "solution": "Use light traps, systemic insecticides, remove affected leaves."
    },
    "Healthy": {
        "cause": "No disease detected. The leaf appears healthy.",
        "solution": "Maintain regular watering, fertilization, and pest monitoring."
    },
    "Powdery Mildew": {
        "cause": "Fungal spores causing white powdery coating on leaves.",
        "solution": "Spray sulfur fungicides, avoid high humidity, prune crowded branches."
    },
    "Sooty Mould": {
        "cause": "Black fungus growing on honeydew secreted by pests like aphids.",
        "solution": "Control pests using neem oil; wash leaves with mild soap solution."
    }
}

# 👉 CLASS_NAMES MUST match DISEASE_INFO ORDER
CLASS_NAMES = list(DISEASE_INFO.keys())

# =====================================================
# 🔧 4. Preprocessing Function (Corrected)
# =====================================================
def preprocess_image(image):
    image = image.convert("RGB")           # Ensure RGB
    image = image.resize((224, 224))       # Ensure correct input size
    img_array = np.array(image) / 255.0    # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# =====================================================
# 🖼️ 5. Image Upload
# =====================================================
uploaded_file = st.file_uploader("📸 Upload an image (Leaf or Fruit)...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = preprocess_image(image)

    # =====================================================
    # 🔮 6. Make Prediction
    # =====================================================
    try:
        prediction = model.predict(img_array)
        predicted_idx = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(np.max(prediction) * 100)

        st.subheader("🔍 Prediction Result")
        st.success(f"**{predicted_class}**")

        # Cause
        st.write(f"### 🔍 Why This Happens\n{DISEASE_INFO[predicted_class]['cause']}")

        # Solution
        st.write(f"### 🌱 Solution\n{DISEASE_INFO[predicted_class]['solution']}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

