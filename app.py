import os
import base64
from io import BytesIO
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


CLASS_NAMES = {
    0: "NORM (Normal)",
    1: "MI (Myocardial Infarction)",
    2: "STTC (ST/T Change)",
    3: "CD (Conduction Disturbance)",
    4: "HYP (Hypertrophy)"
}


model = None
test_data = None
demo_data = None



AGE_MEAN = None     
AGE_STD = None    

SEX_MEAN = None    # Replace with scaler.mean_[1]
SEX_STD = None     # Replace with scaler.scale_[1]

HEIGHT_MEAN = None # Replace with scaler.mean_[2]
HEIGHT_STD = None  # Replace with scaler.scale_[2]

# Option 2: Auto-calculate from current data (approximation)
def calculate_denormalization_params():
    """Calculate approximate denormalization parameters from current data"""
    global AGE_MEAN, AGE_STD, SEX_MEAN, SEX_STD, HEIGHT_MEAN, HEIGHT_STD
    
    if demo_data is not None and len(demo_data) > 0:
        print("📊 Auto-calculating denormalization parameters...")
        print("⚠️ Warning: Using reasonable medical defaults. For accuracy, provide actual scaler parameters.")
        
        # Use reasonable medical ranges as fallback
        AGE_MEAN = 54.0    # Average adult age
        AGE_STD = 16.0     # Reasonable age variation
        
        SEX_MEAN = 0.5     # 50/50 male/female distribution  
        SEX_STD = 0.5      # Binary variable std
        
        HEIGHT_MEAN = 170.0  # Average height in cm
        HEIGHT_STD = 10.0    # Reasonable height variation
        
        print(f"   Using estimated values:")
        print(f"   Age: mean={AGE_MEAN}, std={AGE_STD}")
        print(f"   Sex: mean={SEX_MEAN}, std={SEX_STD}")
        print(f"   Height: mean={HEIGHT_MEAN}, std={HEIGHT_STD}")
        print("   💡 To get exact values, run: python extract_scaler_parameters.py")
    
    return AGE_MEAN, AGE_STD, SEX_MEAN, SEX_STD, HEIGHT_MEAN, HEIGHT_STD

# Option 3: Load scaler directly if available
def load_scaler_if_available():
    """Try to load the scaler object if it exists"""
    import joblib
    import pickle
    global AGE_MEAN, AGE_STD, SEX_MEAN, SEX_STD, HEIGHT_MEAN, HEIGHT_STD
    
    scaler_paths = ['scaler.joblib', 'scaler.pkl', 'model/scaler.joblib', 'model/scaler.pkl']
    
    for path in scaler_paths:
        try:
            scaler = joblib.load(path)
            AGE_MEAN, SEX_MEAN, HEIGHT_MEAN = scaler.mean_
            AGE_STD, SEX_STD, HEIGHT_STD = scaler.scale_
            print(f"✅ Loaded scaler parameters from: {path}")
            print(f"   Age: mean={AGE_MEAN:.3f}, std={AGE_STD:.3f}")
            print(f"   Sex: mean={SEX_MEAN:.3f}, std={SEX_STD:.3f}")
            print(f"   Height: mean={HEIGHT_MEAN:.3f}, std={HEIGHT_STD:.3f}")
            return True
        except:
            try:
                with open(path, 'rb') as f:
                    scaler = pickle.load(f)
                AGE_MEAN, SEX_MEAN, HEIGHT_MEAN = scaler.mean_
                AGE_STD, SEX_STD, HEIGHT_STD = scaler.scale_
                print(f"✅ Loaded scaler parameters from: {path}")
                return True
            except:
                continue
    
    return False

def load_model_and_data():
    """Load model and data with proper error handling"""
    global model, test_data, demo_data
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'model', 'ecg_classifier.keras')
        test_data_path = os.path.join(base_dir, 'data', 'test_ecg.npy')
        demo_data_path = os.path.join(base_dir, 'data', 'test_demo.npy')
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data file not found: {test_data_path}")
        if not os.path.exists(demo_data_path):
            raise FileNotFoundError(f"Demo data file not found: {demo_data_path}")
        
        model = load_model(model_path)
        test_data = np.load(test_data_path)
        demo_data = np.load(demo_data_path)
        
        # Try to load scaler parameters automatically
        if not load_scaler_if_available():
            # Fall back to reasonable defaults if scaler not found
            if AGE_MEAN is None or AGE_STD is None:
                calculate_denormalization_params()
        
        print("✅ Model and data loaded successfully!")
        print(f"   Model inputs: {len(model.inputs) if hasattr(model, 'inputs') else 'Single input'}")
        print(f"   Test data shape: {test_data.shape}")
        print(f"   Demo data shape: {demo_data.shape}")
        print(f"   Number of patients: {len(test_data)}")
        print(f"   Denormalization: Z-score (mean, std) for Age, Sex, Height")
        
    except Exception as e:
        print(f"❌ Error loading model or data: {e}")
        model = None
        test_data = None
        demo_data = None

def generate_ecg_plot(ecg_data, patient_id):
    """Generate ECG plot for all 12 leads with improved styling"""
    try:
        fig, axes = plt.subplots(4, 3, figsize=(15, 10))
        fig.suptitle(f'ECG Record #{patient_id} - 12 Lead Analysis', fontsize=16, fontweight='bold')
        
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        for i in range(12):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Plot ECG signal
            ax.plot(ecg_data[:, i], color='#2E86AB', linewidth=1.2)
            ax.set_title(f'Lead {leads[i]}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            ax.set_ylabel('Amplitude (mV)', fontsize=10)
            
            # Set consistent y-axis limits for better comparison
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close(fig)  # Important: close figure to free memory
        buf.seek(0)
        
        return base64.b64encode(buf.read()).decode('utf-8')
        
    except Exception as e:
        print(f"Error generating ECG plot: {e}")
        # Return a placeholder or empty string
        return ""

@app.route('/')
def home():
    """Home page route"""
    return render_template('home.html')

@app.route('/about')
def about():
    """About page route"""
    return render_template('about.html')

@app.route('/viewer', methods=['GET', 'POST'])
def viewer():
    """ECG viewer and prediction route"""
    max_id = len(test_data) - 1 if test_data is not None else 0
    
    if request.method == 'POST':
        try:
            patient_id_str = request.form.get('patient_id', '').strip()
            
            # Validate input
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
            
            # Check if model and data are loaded
            if model is None or test_data is None or demo_data is None:
                return render_template('viewer.html', 
                                      error="Model or data files not loaded. Please check if all required files exist.",
                                      max_id=max_id)
            
            # Validate patient ID range
            if patient_id < 0 or patient_id > max_id:
                return render_template('viewer.html', 
                                      error=f"Invalid Patient ID. Please enter a number between 0 and {max_id}",
                                      max_id=max_id)
            
            # Get ECG and demographic data
            single_ts = test_data[patient_id]
            single_demo = demo_data[patient_id]
            
            # Validate data shapes
            if single_ts.shape[0] == 0 or single_demo.shape[0] == 0:
                return render_template('viewer.html', 
                                      error=f"No data available for patient ID {patient_id}",
                                      max_id=max_id)
            
            # Prepare input for model (ensure correct shape)
            single_ts_batch = np.expand_dims(single_ts, axis=0)
            single_demo_batch = np.expand_dims(single_demo, axis=0)
            
            # Make prediction
            try:
                prediction = model.predict([single_ts_batch, single_demo_batch], verbose=0)
                
                # Handle different prediction output formats
                if isinstance(prediction, list):
                    prediction = prediction[0]
                
                confidence = float(np.max(prediction[0]))
                class_idx = int(np.argmax(prediction[0]))
                predicted_class = CLASS_NAMES.get(class_idx, f"Unknown Class {class_idx}")
                
            except Exception as pred_error:
                return render_template('viewer.html', 
                                      error=f"Prediction error: {str(pred_error)}",
                                      max_id=max_id)
            
            # Prepare class probabilities
            class_probs = []
            for i, prob in enumerate(prediction[0]):
                class_probs.append({
                    "name": CLASS_NAMES.get(i, f"Class {i}"),
                    "percent": f"{float(prob)*100:.2f}%",
                    "width": float(prob) * 100  # For progress bar
                })
            
            # Sort by probability (highest first)
            class_probs.sort(key=lambda x: x['width'], reverse=True)
            
            # Generate ECG plot
            ecg_plot = generate_ecg_plot(single_ts, patient_id)
            
            # Extract patient info with Z-score denormalization
            try:
                if len(single_demo) >= 3:
                    # Age: denormalize using Z-score formula: actual = (normalized * std) + mean
                    normalized_age = float(single_demo[0])
                    actual_age = (normalized_age * AGE_STD) + AGE_MEAN
                    age = f"{actual_age:.0f} years"
                    
                    # Sex: denormalize and round to get 0 or 1
                    normalized_sex = float(single_demo[1])
                    actual_sex = (normalized_sex * SEX_STD) + SEX_MEAN
                    sex_value = round(actual_sex)  # Round to nearest integer (0 or 1)
                    gender = "Male" if sex_value == 0 else "Female"
                    
                    # Height: denormalize using Z-score formula
                    normalized_height = float(single_demo[2])
                    actual_height = (normalized_height * HEIGHT_STD) + HEIGHT_MEAN
                    height = f"{actual_height:.1f} cm"
                    
                    print(f"Patient {patient_id} info - Age: {age}, Gender: {gender}, Height: {height}")
                    print(f"   Raw normalized values: Age={normalized_age:.3f}, Sex={normalized_sex:.3f}, Height={normalized_height:.3f}")
                    
                else:
                    age = "Unknown"
                    gender = "Unknown"
                    height = "Unknown"
                    print(f"Warning: Incomplete demographic data for patient {patient_id}")
                    
            except (IndexError, ValueError) as e:
                print(f"Error extracting patient info for patient {patient_id}: {e}")
                age = "Unknown"
                gender = "Unknown"
                height = "Unknown"
            
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
            print(f"Unexpected error in viewer route: {e}")
            return render_template('viewer.html', 
                                  error=f"An unexpected error occurred: {str(e)}",
                                  max_id=max_id)
    
    # For GET requests
    return render_template('viewer.html', max_id=max_id)

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500

# Initialize model and data when app starts
load_model_and_data()

if __name__ == '__main__':
    print("🚀 Starting CardioAI Flask Application...")
    print(f"   Model loaded: {'✅' if model is not None else '❌'}")
    print(f"   Data loaded: {'✅' if test_data is not None else '❌'}")
    if test_data is not None:
        print(f"   Available patient IDs: 0 to {len(test_data)-1}")
        print(f"   Demo data format: Age, Sex, Height (all Z-score normalized)")
        print(f"   Denormalization params:")
        print(f"     Age: mean={AGE_MEAN}, std={AGE_STD}")
        print(f"     Sex: mean={SEX_MEAN}, std={SEX_STD} (0=Male, 1=Female)")
        print(f"     Height: mean={HEIGHT_MEAN}, std={HEIGHT_STD}")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)