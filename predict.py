from tensorflow import keras
import cv2
import numpy as np

class Predict_image:
    def __init__(self, image_path):
        self.image_path = image_path

    def load_and_preprocess(self):
        try:
            img_size = 200
            img_array = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            img_array = cv2.resize(img_array, (img_size, img_size))
            img_array = img_array/255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            return str(e)

    def predict_image_class(self, model):
        try:
            labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
            label_map = {label: i for i, label in enumerate(labels)}

            img_array = self.load_and_preprocess()
            prediction = model.predict(img_array)
            prediction_class_index = np.argmax(prediction, axis=1)[0]
            prediction_class_label = list(label_map.keys())[list(label_map.values()).index(prediction_class_index)]
            return prediction_class_label
        except Exception as e:
            return str(e)
    
    def predict(self):
        try:
            # loading model
            model = keras.models.load_model('model.h5')
            # getting resultent class
            predicted_class = self.predict_image_class(model)
            return predicted_class
        except Exception as e:
            return str(e)
        
if __name__ == "__main__":
    image_path = "C:\\Users\\Ajay\\Chatbot\\Testing\\glioma\\Te-gl_0010.jpg"
    model = Predict_image(image_path=image_path)
    print(model.predict())