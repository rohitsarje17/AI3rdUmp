from keras.models import load_model
import cv2
import numpy as np

class NoBallModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        no_ball_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame for the model
            processed_frame = self.preprocess_frame(frame)

            # Make prediction
            prediction = self.model.predict(np.expand_dims(processed_frame, axis=0))

            # Assuming the model outputs a probability for no ball
            if prediction[0][0] > 0.5:  # Threshold can be adjusted
                no_ball_detected = True
                break

        cap.release()
        return no_ball_detected

    def preprocess_frame(self, frame):
        # Resize and normalize the frame for the model
        frame = cv2.resize(frame, (224, 224))  # Adjust size as per model requirement
        frame = frame / 255.0  # Normalize to [0, 1]
        return frame

    def is_no_ball(self, video_path):
        return self.process_video(video_path)