from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from classifier import TextClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Инициализация классификатора
classifier = TextClassifier('model')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Чтение текста из файла
            text = file.read().decode('utf-8')
            
            # Классификация текста
            result = classifier.predict(text)
            
            return render_template('index.html', 
                                 result=result,
                                 filename=file.filename,
                                 text=text[:500] + "..." if len(text) > 500 else text)
    
    return render_template('index.html', result=None)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', debug=True)