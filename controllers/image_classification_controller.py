from fastcore.all import *
from fastai.vision.all import *

# This will be implemented after the ClassificationModel is created
class ImageClassificationController:
    def __init__(self, model_path):
        print('*** init ImageClassificationController')
        self.learn = load_learner(model_path)
    
    def classify_image(self, image_path):
        print('*** Classify Image')
        image_type, _, probs = self.learn.predict(PILImage.create(image_path))
        print(image_path)
        print(f"This is a: {image_type}.")
        print(f"Probability: {probs[0]:.4f}")
        
        # Create a dictionary with the classification result and image path
        # Adjust the path for the src attribute in the <img> tag
        relative_path = os.path.join('static/uploaded', os.path.basename(image_path))
        result = {
            "result": f"This is a {image_type} image with a probability of: {probs[0]:.4f}",
            "src": relative_path
        }
        return result
