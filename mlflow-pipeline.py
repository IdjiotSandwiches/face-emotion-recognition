import dagshub
import mlflow
import numpy as np
import os
import cv2 as cv

class LoadModel:
    def __init__(self):
        self.tracking_url = 'https://dagshub.com/IdjiotSandwiches/face-emotion-recognition.mlflow'
        self.model_name = 'ResNet50_FER'
        self.model_version = 'latest'
        self.model_uri = f'models:/{self.model_name}/{self.model_version}'
        self.repo_owner = 'IdjiotSandwiches'
        self.repo_name = 'face-emotion-recognition'

    def set_mlflow_tracking_url(self):
        dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name, mlflow=True)
        mlflow.set_tracking_uri(self.tracking_url)

    def load(self):
        self.model = mlflow.pyfunc.load_model(self.model_uri)
    
    def predict(self, image):
        predict = self.model.predict(image)
        idx = np.argmax(predict)

        return idx

class PreprocessImage:
    def __init__(self):
        self.img_size = (224,224)

    def preprocess(self, image):
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.image.resize(image, self.img_size)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.expand_dims(image, axis=0).numpy()

        return image

class RealTimeCamera:
    def __init__(self):
        self.face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    def capture_image(self, model, preprocess_image):
        capture = cv.VideoCapture(0)
        while True:
            _, frame = capture.read()
            labels = []

            img = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            detected = self.face_classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

            for rect in detected:
                x,y,w,h = rect
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
                
                img = img[y:y+h, x:x+w]

                img = preprocess_image.preprocess(img)
                prediction = model.predict(img)
                label = self.labels[prediction]

                cv.putText(frame, label, (x,y-10), cv.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 1)
            
            cv.imshow('Emotion', frame)


if __name__ == '__main__':
    def load_model():
        model = LoadModel.set_mlflow_tracking_url()
        model = LoadModel.load()
        return model

    model = load_model()
    preprocess = PreprocessImage
    real_time_camera = RealTimeCamera.capture_image(model=model, preprocess_image=preprocess)
