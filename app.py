from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
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
# 2. FLASK ROUTES
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

        # Preprocessing (224x224 & Rescaling)
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  

        # Prediction Logic
        preds = model.predict(img_array)
        pred_class_index = int(np.argmax(preds[0])) 
        final_class = CLASS_LABELS[pred_class_index]
        confidence = round(100 * float(np.max(preds[0])), 2)

        # Hamesha index.html render karega, raw data nahi dega
        return render_template('index.html', 
                               prediction=final_class, 
                               confidence=f"{confidence}%", 
                               user_image=filepath)

    except Exception as e:
        print(f"FLASK ERROR: {str(e)}")
        return render_template('index.html', error='Internal Server Error. Please check terminal for details.')

if __name__ == '__main__':
    app.run(debug=True)