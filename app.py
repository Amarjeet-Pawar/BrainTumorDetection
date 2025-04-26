import os
from flask import Flask, render_template, request
from predictor import check, model  # Import the model and check function
import torch

author = 'TEAM'

app = Flask(__name__)

# Create directories if they don't exist
UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set device (either 'cpu' or 'cuda')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move the model to the correct device

@app.route('/')
@app.route('/index')
def index():
    return render_template('Upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('Upload.html', error='No file selected')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('Upload.html', error='No file selected')

        if file:
            # Save the file
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Get prediction
            pred_status, confidence = check(filepath, model, device)
            
            # Convert confidence to percentage and ensure it's a float
            try:
                confidence_pct = float(confidence) * 100
            except (TypeError, ValueError):
                confidence_pct = 0.0  # Default to 0 if conversion fails

            return render_template('complete.html',
                                image_name=filename,
                                predvalue=bool(pred_status),  # Ensure boolean value
                                pred_prob=confidence_pct)  # Confidence score as percentage

    return render_template('Upload.html')

if __name__ == "__main__":
    app.run(port=4555, debug=True)
