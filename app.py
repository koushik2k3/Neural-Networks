from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
# from lib import * 
from libv2 import *
from libv3 import *
import os
import pandas as pd
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'

# Set the path for image uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class UploadForm(FlaskForm):
    image = FileField('Image')
    submit = SubmitField('Upload')

W1 = np.array(pd.read_excel("weights1.xlsx"))
W2 = np.array(pd.read_excel("weights2.xlsx"))
b1 = np.array(pd.read_excel("bias1.xlsx"))
b2 = np.array(pd.read_excel("bias2.xlsx"))

w1 = np.array(pd.read_excel("weights.xlsx"))

# Set up the upload folder for storing images
DRAW_UPLOAD_FOLDER = 'static/drawings'
app.config['DRAW_UPLOAD_FOLDER'] = DRAW_UPLOAD_FOLDER
os.makedirs(DRAW_UPLOAD_FOLDER, exist_ok=True)

network = [
    Convolutional((1, 28, 28), 3, 5),
    Activation(sigmoid, sigmoid_gradient),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Activation(sigmoid, sigmoid_gradient),
    Dense(100, 10),
    Softmax()
]

ct = 1
for i in network:
    try:
        i.load_weights("weights_"+str(ct)+".npz")
    except:
        pass
    ct+=1



@app.route('/', methods=['GET', 'POST'])
def project():
    form = UploadForm()
    image_path = None
    output = None
    
    if form.validate_on_submit():
        try:
            image_file = request.files['image']
            image_file.save('static/images/' + image_file.filename)
            image_path = image_file.filename
        except:
            return render_template('index.html', form=form, image=image_path, output=output)

    if image_path:
        # output = test_prediction('static/images/'+image_path, W1, b1, W2, b2)
        image = Image.open('static/images/'+image_path)
        image = image.convert("L").resize((28, 28))

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Reshape the numpy array to (1, 28, 28)
        image_array = image_array.reshape(1, 28, 28)
        output = predict_final(network, image_array)
        output = class_names[np.argmax(output)]

        # if output%2==0:
        #     output = str(output) + ' Even'
        # else:
        #     output = str(output) + ' Odd'

    return render_template('index.html', form=form, image=image_path, output=output)

@app.route('/draw', methods=['GET', 'POST'])
def draw_number():
    if request.method == 'POST':
        input_number = int(request.form['input_number'])
        if input_number<0:
            return render_template('index2.html')

        try:
            draw(input_number, w1)
        except:
            drawmorethan1digit(input_number, w1)
        
        image_filename = f'image_{input_number}.png'
        
        return render_template('index2.html', image_filename=image_filename)
    
    return render_template('index2.html')

@app.route('/static/drawings/<filename>')
def display_image(filename):
    return send_from_directory(app.config['DRAW_UPLOAD_FOLDER'], 'drawing.png')

if __name__ == '__main__':
    app.run(port=4000,debug=True)
