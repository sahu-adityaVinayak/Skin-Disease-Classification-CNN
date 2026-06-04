from flask import Flask, render_template, request
import os
import urllib.request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ==========================================
# 1. MODEL LOADING (auto-download from GitHub Release)
# ==========================================
MODEL_URL = 'https://github.com/sahu-adityaVinayak/Skin-Disease-Classification-CNN/releases/download/v1.0/model.h5'
MODEL_PATH = 'model/model.h5'

os.makedirs('model', exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print('Downloading model weights from GitHub Release...')
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print('Model downloaded successfully!')

print('Loading the CNN Model... Please wait.')
model = load_model(MODEL_PATH)
print('Model loaded successfully!')

CLASS_LABELS = {
    0: 'Benign (Safe / Harmless)',
    1: 'Malignant (Dangerous / Consult Doctor)'
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_skin_image(img_path):
    import PIL.Image
    img = PIL.Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    R = img_array[:,:,0].astype(float)
    G = img_array[:,:,1].astype(float)
    B = img_array[:,:,2].astype(float)
    skin_mask = (R > 95) & (G > 40) & (B > 20) & (R > G) & (R > B) & (np.abs(R - G) > 15)
    skin_ratio = np.sum(skin_mask) / (img_array.shape[0] * img_array.shape[1])
    return skin_ratio > 0.2

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded.')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected.')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        if not is_valid_skin_image(filepath):
            return render_template('result.html', prediction='Invalid Image', confidence=0,
                                   image_path=filename, error='Not a valid skin image.')
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        preds = model.predict(img_array)
        pred_class_index = int(np.argmax(preds[0]))
        final_class = CLASS_LABELS[pred_class_index]
        confidence = round(100 * float(np.max(preds[0])), 2)
        return render_template('result.html', prediction=final_class,
                               confidence=confidence, image_path=filename)
    return render_template('index.html', error='Invalid file type.')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
