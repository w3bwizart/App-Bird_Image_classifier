from fastcore.all import *
from fastai.vision.all import *

class NeuralNetworkController:
    def train_model(self, path):
        print('*** Train Model')
        dls = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            get_y=parent_label,
            item_tfms=Resize(192, method='squish')
        ).dataloaders(path)

        learn = vision_learner(dls, resnet18, metrics=error_rate)
        learn.fine_tune(3)
        
         # Save the trained model
        learn.export('model.pkl')
        print('*** Model Trained ')
        return 'OK'
        