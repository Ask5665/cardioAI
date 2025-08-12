import os
import base64
from io import BytesIO
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
import gdown
import logging
warnings.filterwarnings('ignore')

# Configure logging for Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Render-specific Configuration
# -------------------------------
app = Flask(__name__)

# Get port from environment variable (Render sets this)
PORT = int(os.environ.get('PORT', 5000))

# Configure matplotlib for server environment
plt.ioff()  # Turn off interactive plotting

# -------------------------------
# Google Drive Auto Download Setup
# -------------------------------
FOLDER_URL = "https://drive.google.com/drive/folders/1bwA-s-l-6bD8WOPv8rDOZptbpP-t9veZ?usp=sharing"
NEEDED_FILES = [
    os.path.join("model", "ecg_classifier.keras"),
    os.path.join("data", "test_ecg.npy"),
    os.path.join("data", "test_demo.npy")
]

def ensure_files_exist():
    """Download model/data from Google Drive only if missing."""
    missing_files = [f for f in NEEDED_FILES if not os.path.exists(f)]
    if missing_files:
        logger.info(f"📥 Missing files detected: {missing_files}")
        logger.info("🔄 Downloading model and data from Google Drive...")
        try:
            # Create directories if they don't exist
            os.makedirs("model", exist_ok=True)
            os.makedirs("data", exist_ok=True)
            
            # Download with error handling for server environment
            gdown.download_folder(
                url=FOLDER_URL, 
                output=os.getcwd(), 
                quiet=False, 
                use_cookies=False,
                remaining_ok=True  # Don't fail if some files already exist
            )
            logger.info("✅ Download complete!")
            
            # Verify files were downloaded
            still_missing = [f for f in NEEDED_FILES if not os.path.exists(f)]
            if still_missing:
                logger.warning(f"⚠️  Warning: Some files still missing after download: {still_missing}")
                return False
            return True
        except Exception as e:
            logger.error(f"❌ Error downloading files: {e}")
            return False
    else:
        logger.info("✅ Model and data already exist — skipping download.")
        return True

CLASS_NAMES = {
    0: "NORM (Normal)",
    1: "MI (Myocardial Infarction)",
    2: "STTC (ST/T Change)",
    3: "CD (Conduction Disturbance)",
    4: "HYP (Hypertrophy)"
}

# Global variables
model = None
test_data = None
demo_data = None

# Denormalization parameters
AGE_MEAN = 54.0
AGE_STD = 16.0
SEX_MEAN = 0.5
SEX_STD = 0.5
HEIGHT_MEAN = 170.0
HEIGHT_STD = 10.0

def load_scaler_if_available():
    """Try to load the scaler object if it exists"""
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
    
    logger.info("Using default denormalization parameters")
    return False

def load_model_and_data():
    """Load model and data with proper error handling for server environment"""
    global model, test_data, demo_data
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'model', 'ecg_classifier.keras')
        test_data_path = os.path.join(base_dir, 'data', 'test_ecg.npy')
        demo_data_path = os.path.join(base_dir, 'data', 'test_demo.npy')
        
        # Check if files exist
        if not all(os.path.exists(path) for path in [model_path, test_data_path, demo_data_path]):
            missing = [path for path in [model_path, test_data_path, demo_data_path] if not os.path.exists(path)]
            raise FileNotFoundError(f"Required files not found: {missing}")
        
        # Load model with TensorFlow optimizations for server
        model = keras.models.load_model(model_path, compile=False)
        model.compile()  # Recompile for inference
        
        test_data = np.load(test_data_path)
        demo_data = np.load(demo_data_path)
        
        # Load scaler parameters
        load_scaler_if_available()
        
        logger.info("✅ Model and data loaded successfully!")
        logger.info(f"   Test data shape: {test_data.shape}")
        logger.info(f"   Demo data shape: {demo_data.shape}")
        logger.info(f"   Number of patients: {len(test_data)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model or data: {e}")
        model = None
        test_data = None
        demo_data = None
        return False

def generate_ecg_plot(ecg_data, patient_id):
    """Generate ECG plot optimized for server environment"""
    try:
        # Use smaller figure size for better performance
        fig, axes = plt.subplots(4, 3, figsize=(12, 8))
        fig.suptitle(f'ECG Record #{patient_id} - 12 Lead Analysis', fontsize=14, fontweight='bold')
        
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        for i in range(12):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            ax.plot(ecg_data[:, i], color='#2E86AB', linewidth=1.0)
            ax.set_title(f'Lead {leads[i]}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            ax.set_ylabel('mV', fontsize=8)
            
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save with optimized settings for server
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', optimize=True)
        plt.close(fig)  # Critical: close figure to free memory
        buf.seek(0)
        
        return base64.b64encode(buf.read()).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Error generating ECG plot: {e}")
        return ""

@app.route('/')
def home():
    """Home page route"""
    return render_template('home.html')

@app.route('/about')
def about():
    """About page route"""
    return render_template('about.html')

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_loaded': test_data is not None and demo_data is not None,
        'patient_count': len(test_data) if test_data is not None else 0
    }
    return status

@app.route('/viewer', methods=['GET', 'POST'])
def viewer():
    """ECG viewer and prediction route"""
    max_id = len(test_data) - 1 if test_data is not None else 0
    
    if request.method == 'POST':
        try:
            patient_id_str = request.form.get('patient_id', '').strip()
            
            if not patient_id_str:
                return render_template('viewer.html', 
                                      error="Please enter a patient ID",
                                      max_id=max_id)
            
            try:
                patient_id = int(patient_id_str)
            except ValueError:
                return render_template('viewer.html', 
                                      error="Please enter a valid number",
                                      max_id=max_id)
            
            if model is None or test_data is None or demo_data is None:
                return render_template('viewer.html', 
                                      error="Model or data files not loaded. Please try again later.",
                                      max_id=max_id)
            
            if patient_id < 0 or patient_id > max_id:
                return render_template('viewer.html', 
                                      error=f"Invalid Patient ID. Please enter a number between 0 and {max_id}",
                                      max_id=max_id)
            
            # Get data
            single_ts = test_data[patient_id]
            single_demo = demo_data[patient_id]
            
            # Prepare for prediction
            single_ts_batch = np.expand_dims(single_ts, axis=0)
            single_demo_batch = np.expand_dims(single_demo, axis=0)
            
            # Make prediction
            try:
                prediction = model.predict([single_ts_batch, single_demo_batch], verbose=0)
                
                if isinstance(prediction, list):
                    prediction = prediction[0]
                
                confidence = float(np.max(prediction[0]))
                class_idx = int(np.argmax(prediction[0]))
                predicted_class = CLASS_NAMES.get(class_idx, f"Unknown Class {class_idx}")
                
            except Exception as pred_error:
                logger.error(f"Prediction error: {pred_error}")
                return render_template('viewer.html', 
                                      error="Prediction failed. Please try again.",
                                      max_id=max_id)
            
            # Prepare class probabilities
            class_probs = []
            for i, prob in enumerate(prediction[0]):
                class_probs.append({
                    "name": CLASS_NAMES.get(i, f"Class {i}"),
                    "percent": f"{float(prob)*100:.2f}%",
                    "width": float(prob) * 100
                })
            
            class_probs.sort(key=lambda x: x['width'], reverse=True)
            
            # Generate plot
            ecg_plot = generate_ecg_plot(single_ts, patient_id)
            
            # Extract patient info
            try:
                if len(single_demo) >= 3:
                    normalized_age = float(single_demo[0])
                    actual_age = (normalized_age * AGE_STD) + AGE_MEAN
                    age = f"{actual_age:.0f} years"
                    
                    normalized_sex = float(single_demo[1])
                    actual_sex = (normalized_sex * SEX_STD) + SEX_MEAN
                    sex_value = round(actual_sex)
                    gender = "Male" if sex_value == 0 else "Female"
                    
                    normalized_height = float(single_demo[2])
                    actual_height = (normalized_height * HEIGHT_STD) + HEIGHT_MEAN
                    height = f"{actual_height:.1f} cm"
                else:
                    age = gender = height = "Unknown"
            except Exception:
                age = gender = height = "Unknown"
            
            return render_template('viewer.html', 
                                  patient_id=patient_id,
                                  age=age,
                                  gender=gender,
                                  height=height,
                                  predicted_class=predicted_class,
                                  confidence=f"{confidence*100:.2f}%",
                                  class_probs=class_probs,
                                  ecg_plot=ecg_plot,
                                  max_id=max_id)
            
        except Exception as e:
            logger.error(f"Unexpected error in viewer: {e}")
            return render_template('viewer.html', 
                                  error="An unexpected error occurred. Please try again.",
                                  max_id=max_id)
    
    return render_template('viewer.html', max_id=max_id)

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return render_template('500.html'), 500

def initialize_app():
    """Initialize the application for server deployment"""
    logger.info("🚀 Initializing CardioAI Flask Application for Render...")
    
    try:
        # Step 1: Ensure files exist
        if not ensure_files_exist():
            logger.error("❌ Failed to download required files")
            return False
        
        # Step 2: Load model and data
        if not load_model_and_data():
            logger.error("❌ Failed to load model and data")
            return False
        
        logger.info("✅ Application initialized successfully for Render!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False

# Initialize the app
initialize_app()

if __name__ == '__main__':
    # For local development
    app.run(debug=False, host='0.0.0.0', port=PORT)
else:
    # For Render deployment
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
