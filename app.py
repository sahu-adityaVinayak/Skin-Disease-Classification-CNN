from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ==========================================
# 1. MODEL LOADING 
# ==========================================
MODEL_PATH = 'model/model.h5' 
print("Loading the CNN Model... Please wait.")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

CLASS_LABELS = {
    0: 'Benign (Safe / Harmless)',
    1: 'Malignant (Dangerous / Consult Doctor)'
}

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==========================================
# 2. THE GATEKEEPER: SKIN DETECTION FILTER
# ==========================================
def is_valid_skin_image(image_path):
    """
    Mathematical Heuristic Filter (Kovac's Rule) to detect if the image has skin tones.
    Prevents out-of-distribution (OOD) images like screenshots or random objects.
    """
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        
        # Extract color channels
        R = img_array[:,:,0]
        G = img_array[:,:,1]
        B = img_array[:,:,2]
        
        # Scientific RGB rules for human skin
        rule1 = (R > 95) & (G > 40) & (B > 20)
        rule2 = (R > G) & (R > B)
        
        max_c = np.maximum(np.maximum(R, G), B)
        min_c = np.minimum(np.minimum(R, G), B)
        rule3 = (max_c - min_c) > 15
        
        # Calculate percentage of skin pixels
        skin_mask = rule1 & rule2 & rule3
        skin_percentage = np.mean(skin_mask) * 100
        
        # At least 10% of the image must contain skin colors
        return skin_percentage > 10.0
    except Exception as e:
        print(f"Skin Filter Error: {e}")
        return True # Fallback if error occurs

# ==========================================
# 3. FLASK ROUTES
# ==========================================
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/analyze', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded. Please select an image.')
            
        f = request.files['file']
        if f.filename == '':
            return render_template('index.html', error='No file selected. Please click on the box to choose an image.')

        # Save Image
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)

        # ------------------------------------------
        # SECURITY CHECK: Pass through Gatekeeper
        # ------------------------------------------
        if not is_valid_skin_image(filepath):
            return render_template('index.html', 
                                   prediction="Invalid Image (No Skin Detected)", 
                                   confidence="N/A", 
                                   user_image=filepath,
                                   error="Security Alert: Our pre-filter detected that this is not a valid skin image. Please upload a proper dermoscopic photo.")

        # Preprocessing for MobileNet
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  

        # Prediction Logic
        preds = model.predict(img_array)
        pred_class_index = int(np.argmax(preds[0])) 
        final_class = CLASS_LABELS[pred_class_index]
        confidence = round(100 * float(np.max(preds[0])), 2)

        return render_template('index.html', 
                               prediction=final_class, 
                               confidence=f"{confidence}%", 
                               user_image=filepath)

    except Exception as e:
        print(f"FLASK ERROR: {str(e)}")
        return render_template('index.html', error='Internal Server Error. Please check terminal for details.')

if __name__ == '__main__':
    app.run(debug=True)