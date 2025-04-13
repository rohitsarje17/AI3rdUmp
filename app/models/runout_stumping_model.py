from flask import current_app
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO  # Correct way to load models now
from roboflow import Roboflow

class RunoutStumpingModel:
    def __init__(self):
        stumps_model_path = 'E:\\BTech\\Project\\third-umpire-decision\\trainedModels\\stumps.pt'
        batsman_model_path = 'E:\\BTech\\Project\\third-umpire-decision\\trainedModels\\insideOutside.pt'

        # Use the latest YOLO format
        self.stumps_model = YOLO(stumps_model_path)
        self.batsman_model = YOLO(batsman_model_path)

        # rf = Roboflow(api_key="eGBCVKyNP2NYRQpzleUF")
        # project = rf.workspace("rohitsarje").project("insideoutside")
        # version = project.version(4)
        # self.batsman_model = version.model

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        is_out = False
        frame_count = 0
        hit_frame = None
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if self.detect_stumps(frame):
                hit_frame = frame
                print(f"Stumps hit detected at frame {frame_count}")
                break

            frame_filename = f"frame_{frame_count}.jpg"
            frame_path = current_app.config['STATIC_FOLDER'] + '/' + frame_filename
            cv2.imwrite(frame_path, frame)
            frames.append({'filename': frame_filename, 'label': f"Frame {frame_count}"})

        cap.release()

        if hit_frame is not None:
            hit_frame_rgb = cv2.cvtColor(hit_frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(current_app.config['UPLOAD_FOLDER'] + '/hit_frame.jpg', hit_frame)
            cv2.imwrite(current_app.config['STATIC_FOLDER'] + '/hit_frame.jpg', hit_frame)
            plt.imshow(hit_frame_rgb)
            plt.savefig(current_app.config['STATIC_FOLDER'] + '/hit_frame_plot.jpg')  # Save the plot instead of showing it

            if not self.check_batsman_in_crease(hit_frame):
                is_out = True

            frames.append({'filename': 'hit_frame.jpg', 'label': 'Hit Frame'})  # Ensure hit_frame is the last frame

        return is_out, frames

    def detect_stumps(self, frame):
        results = self.stumps_model(frame)
        for r in results:
            for box in r.boxes:
                if int(box.cls) == 1:  
                    return True
        return False

    def check_batsman_in_crease(self, frame):
        results = self.batsman_model(frame)
        for r in results:
            for box in r.boxes:
                if int(box.cls) == 1:  
                    return False
        return True
