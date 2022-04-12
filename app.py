import base64
from flask import Flask, render_template, request
from inference import prediction, model_info
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.form.get('file')
        if not file:
            return "No file received"
        img_bytes = base64.b64decode(file)
        return prediction(img_bytes)

    return render_template('index.html')

@app.route('/model_info', methods=['GET'])
def model_info_page():
    info = model_info()
    return render_template('model_info.html', model_name=info['model_name'], train_acc=info['train_accuracy'], val_acc=info['val_accuracy'])
