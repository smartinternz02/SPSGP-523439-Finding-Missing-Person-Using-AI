from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('Missing.h5')
name = ["Found Missing", "Normal"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['image']
    img = Image.open(image)
    img = img.resize((64, 64))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    pred_class = np.argmax(pred, axis=1)[0]
    result = name[pred_class]
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
