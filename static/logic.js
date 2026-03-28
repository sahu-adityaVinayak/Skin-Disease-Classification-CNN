const fileInput = document.getElementById('file-input');
const preview = document.getElementById('image-preview');
const uploadText = document.getElementById('upload-text');
const analyzeBtn = document.getElementById('analyze-btn');
const resultArea = document.getElementById('result-area');
const loader = document.getElementById('loader');

// Image Preview logic
fileInput.addEventListener('change', function() {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
            uploadText.style.display = 'none';
            analyzeBtn.disabled = false;
        }
        reader.readAsDataURL(file);
    }
});

// Send image to Flask Backend
analyzeBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    // Update UI to loading state
    analyzeBtn.style.display = 'none';
    loader.style.display = 'block';
    resultArea.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
        } else {
            // Show real AI results
            document.getElementById('prediction-text').innerHTML = `Class: <strong>${data.class}</strong>`;
            document.getElementById('confidence-text').innerHTML = `Confidence: <strong>${data.confidence}</strong>`;
            resultArea.style.display = 'block';
        }
    } catch (error) {
        console.error("Error connecting to server:", error);
        alert("Server error. Make sure app.py is running!");
    } finally {
        loader.style.display = 'none';
        analyzeBtn.style.display = 'block';
    }
});