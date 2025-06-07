from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your model
model = tf.keras.models.load_model("D:/Desktop2/payalcnn/payal_model.h5")


# CIFAR-10 classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog','frog',  'horse', 'ship', 'truck']

# Route for home page
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        img_file = request.files['imagefile']
        if img_file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
            img_file.save(img_path)

            # Preprocess the image
            img = image.load_img(img_path, target_size=(32, 32))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Predict
            predictions = model.predict(img_array)
            probabilities = tf.nn.softmax(predictions)
            predicted_index = tf.argmax(probabilities, axis=1).numpy()[0]
            print("Prediction index:", predicted_index)
            predicted_class = class_names[predicted_index]

            return render_template('index.html',
                                   prediction=predicted_class,
                                   img_path=img_path)

    return render_template('index.html', prediction=None, img_path=None)


if __name__ == '__main__':
    app.run(debug=True)
