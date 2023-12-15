from flask import Flask, render_template, request
from controllers.image_collection_controller import ImageCollectionController
from controllers.neural_network_controller import NeuralNetworkController
from controllers.image_classification_controller import ImageClassificationController
from fastai.vision.all import *
from pathlib import Path
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)
image_collection_controller = ImageCollectionController()
neural_network_controller = NeuralNetworkController()
image_classification_controller = None  # Initialized after the model is trained


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/collect-images', methods=['POST'])
def run_image_classification():
    print('*** Prepare Dataset')
    result = image_collection_controller.run_image_classification()
    return render_template('home.html', result=result)

@app.route('/train-model', methods=['POST'])
def train_model():
    print('*** Start training Model')
    # Path where the images are stored and where the model will be saved
    path_to_data = 'data_set'
    model_name = 'trained_model'

    # Train the model
    neural_network_controller.train_model(path_to_data)

    # Initialize the ImageClassificationController with the path of the trained model
    global image_classification_controller
    model_path = 'model.pkl'
    image_classification_controller = ImageClassificationController(str(model_path))

    return render_template('home.html', result2="Model training done.")







@app.route('/classify', methods=['GET', 'POST'])
def classify_image():
    print('*** Classify Image')
    
    app.config['UPLOAD_FOLDER'] = 'static/uploaded/'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
     
    test_image_path = 'data_set/bird/bird_5.jpg'
    
     # Get the stored model
    image_classification_controller = ImageClassificationController('model.pkl')
    
    def allowed_file(filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            return 'No file part'
        file = request.files['image']
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Perform classification
            if image_classification_controller is None:
                return "Model not trained yet."
            result4 = image_classification_controller.classify_image(filepath)
            return render_template('home.html', result4=result4)
    else:
        # Perform classification
        result3 = image_classification_controller.classify_image(str(test_image_path))
        print(result3)
        return render_template('home.html', result3=result3)


if __name__ == '__main__':
    app.run(debug=True)