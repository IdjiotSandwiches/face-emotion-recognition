import dagshub
import mlflow
import numpy as np
import os
import cv2 as cv
import tensorflow as tf

class LoadModel:
    def __init__(self):
        self.tracking_url = 'https://dagshub.com/IdjiotSandwiches/face-emotion-recognition.mlflow'
        self.model_name = 'ResNet50_FER'
        self.model_version = 'latest'
        self.model_uri = f'models:/{self.model_name}/{self.model_version}'
        self.repo_owner = 'IdjiotSandwiches'
        self.repo_name = 'face-emotion-recognition'
        self.logged_model = 'runs:/8bb2da5217c04301a5b5906e59486d36/CNN_FER'

    def set_mlflow_tracking_url(self):
        dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name, mlflow=True)
        mlflow.set_tracking_uri(self.tracking_url)

    def load(self):
        self.model = mlflow.pyfunc.load_model(self.logged_model)
        mlflow.pyfunc.get_model_dependencies(self.logged_model)
    
    def predict(self, image):
        predict = self.model.predict(image)
        idx = np.argmax(predict)

        return idx

class PreprocessImage:
    def __init__(self):
        self.img_size = (48,48)

    def preprocess(self, image):
        image = tf.image.resize(image, self.img_size)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.expand_dims(image, axis=0).numpy()
        return image

class RealTimeCamera:
    def __init__(self):
        self.face_classifier = cv.CascadeClassifier(cv.data.haarcascades + './haarcascade_frontalface_default.xml')
        self.labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    def capture_image(self, model, preprocess_image):
        capture = cv.VideoCapture(0)
        while True:
            _, frame = capture.read()

            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            detected = self.face_classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

            if(len(detected) < 1):
                continue

            for rect in detected:
                x,y,w,h = rect
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)

                if w == 0 or h == 0:
                    continue
                
                img = img[y:y+h, x:x+w]

                if img.size == 0:
                    continue

                img = preprocess_image.preprocess(img)
                prediction = model.predict(img)
                label = self.labels[prediction]

                cv.putText(frame, label, (x,y-10), cv.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 1)
            
            cv.imshow('Emotion', frame)
            cv.waitKey(1)

if __name__ == '__main__':
    model = LoadModel()
    model.set_mlflow_tracking_url()
    model.load()
    preprocess = PreprocessImage()
    img = cv.imread("C:\\Users\\vinar\\OneDrive\\Pictures\\Screenshots\\Screenshot 2024-12-07 193236.png")
    img = preprocess.preprocess(img)

    idx = model.predict(img)

    real_time = RealTimeCamera()
    labels = real_time.labels[idx]
    print(labels)
    # real_time_camera = RealTimeCamera()
    # real_time_camera.capture_image(model=model, preprocess_image=preprocess)
    