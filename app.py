import os
import base64
from io import BytesIO
import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import gdown
import logging

warnings.filterwarnings('ignore')

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
PORT = int(os.environ.get('PORT', 5000))
plt.ioff()

# Google Drive links
MODEL_URL = "https://drive.google.com/drive/folders/1X2nGjojRO8OvrEOCo8jytBAEV7MN-GYz?usp=sharing"
DATA_URL = "https://drive.google.com/drive/folders/1ftrOsfgVmCneaF5Q44wksMpuXGsO7g14?usp=sharing"

NEEDED_FILES = [
    os.path.join("model", "ecg_classifier.keras"),
    os.path.join("data", "test_ecg.npy"),
    os.path.join("data", "test_demo.npy")
]

CLASS_NAMES = {
    0: "NORM (Normal)",
    1: "MI (Myocardial Infarction)",
    2: "STTC (ST/T Change)",
    3: "CD (Conduction Disturbance)",
    4: "HYP (Hypertrophy)"
}

# Globals
model = None
test_data = None
demo_data = None

# Denormalization defaults
AGE_MEAN = 54.0
AGE_STD = 16.0
SEX_MEAN = 0.5
SEX_STD = 0.5
HEIGHT_MEAN = 170.0
HEIGHT_STD = 10.0


def ensure_files_exist():
    """Ensure model and data exist, download if missing."""
    os.makedirs("model", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    if not os.path.exists("model/ecg_classifier.keras"):
        logger.info("📥 Downloading model files...")
        gdown.download_folder(MODEL_URL, output="model", quiet=False, use_cookies=False)

    if not (os.path.exists("data/test_ecg.npy") and os.path.exists("data/test_demo.npy")):
        logger.info("📥 Downloading data files...")
        gdown.download_folder(DATA_URL, output="data", quiet=False, use_cookies=False)

    # Check again
    missing = [f for f in NEEDED_FILES if not os.path.exists(f)]
    if missing:
        logger.error(f"❌ Still missing files after download: {missing}")
        return False
    logger.info("✅ All required files present.")
    return True


def load_scaler_if_available():
    """Try loading scaler parameters."""
    global AGE_MEAN, AGE_STD, SEX_MEAN, SEX_STD, HEIGHT_MEAN, HEIGHT_STD
    scaler_paths = ['scaler.joblib', 'scaler.pkl', 'model/scaler.joblib', 'model/scaler.pkl']
    for path in scaler_paths:
        try:
            import joblib
            scaler = joblib.load(path)
            AGE_MEAN, SEX_MEAN, HEIGHT_MEAN = scaler.mean_
            AGE_STD, SEX_STD, HEIGHT_STD = scaler.scale_
            logger.info(f"✅ Loaded scaler parameters from: {path}")
            return True
        except:
            continue
    logger.info("ℹ️ Using default denormalization parameters.")
    return False


def load_model_and_data():
    """Load model and data into memory."""
    global model, test_data, demo_data
    try:
        model_path = os.path.join("model", "ecg_classifier.keras")
        test_path = os.path.join("data", "test_ecg.npy")
        demo_path = os.path.join("data", "test_demo.npy")

        model = keras.models.load_model(model_path, compile=False)
        model.compile()
        test_data = np.load(test_path)
        demo_data = np.load(demo_path)

        load_scaler_if_available()
        logger.info("✅ Model and data loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model/data: {e}")
        return False


def generate_ecg_plot(ecg_data, patient_id):
    """Generate ECG plot."""
    try:
        fig, axes = plt.subplots(4, 3, figsize=(12, 8))
        fig.suptitle(f'ECG Record #{patient_id}', fontsize=14, fontweight='bold')
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        for i in range(12):
            row, col = divmod(i, 3)
            ax = axes[row, col]
            ax.plot(ecg_data[:, i], color='#2E86AB', linewidth=1.0)
            ax.set_title(f'Lead {leads[i]}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            ax.set_ylabel('mV', fontsize=8)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Plot error: {e}")
        return ""


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/viewer', methods=['GET', 'POST'])
def viewer():
    max_id = len(test_data) - 1 if test_data is not None else 0
    if request.method == 'POST':
        try:
            patient_id_str = request.form.get('patient_id', '').strip()
            if not patient_id_str.isdigit():
                return render_template('viewer.html', error="Enter a valid patient ID", max_id=max_id)
            patient_id = int(patient_id_str)
            if patient_id < 0 or patient_id > max_id:
                return render_template('viewer.html', error=f"ID must be 0 to {max_id}", max_id=max_id)

            single_ts = test_data[patient_id]
            single_demo = demo_data[patient_id]
            pred = model.predict([np.expand_dims(single_ts, 0), np.expand_dims(single_demo, 0)], verbose=0)[0]
            confidence = np.max(pred)
            class_idx = np.argmax(pred)
            predicted_class = CLASS_NAMES.get(class_idx, f"Class {class_idx}")

            class_probs = [{"name": CLASS_NAMES.get(i, f"Class {i}"),
                            "percent": f"{p*100:.2f}%",
                            "width": p*100} for i, p in enumerate(pred)]
            class_probs.sort(key=lambda x: x["width"], reverse=True)

            age = gender = height = "Unknown"
            if len(single_demo) >= 3:
                age = f"{(single_demo[0] * AGE_STD) + AGE_MEAN:.0f} years"
                gender = "Male" if round((single_demo[1] * SEX_STD) + SEX_MEAN) == 0 else "Female"
                height = f"{(single_demo[2] * HEIGHT_STD) + HEIGHT_MEAN:.1f} cm"

            ecg_plot = generate_ecg_plot(single_ts, patient_id)

            return render_template('viewer.html', patient_id=patient_id, age=age, gender=gender, height=height,
                                   predicted_class=predicted_class, confidence=f"{confidence*100:.2f}%",
                                   class_probs=class_probs, ecg_plot=ecg_plot, max_id=max_id)
        except Exception as e:
            logger.error(f"Viewer error: {e}")
            return render_template('viewer.html', error="An error occurred", max_id=max_id)
    return render_template('viewer.html', max_id=max_id)


@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_loaded': test_data is not None,
        'patient_count': len(test_data) if test_data is not None else 0
    }


def initialize_app():
    logger.info("🚀 Initializing app...")
    if not ensure_files_exist():
        return False
    if not load_model_and_data():
        return False
    return True


initialize_app()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=PORT)
