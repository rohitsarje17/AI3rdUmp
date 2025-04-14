from flask import current_app
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO  # Correct way to load models now
from roboflow import Roboflow
import os
from PIL import Image
import io
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

class RunoutStumpingModel:
    def __init__(self):
        stumps_model_path = 'E:\\BTech\\Project\\third-umpire-decision\\trainedModels\\stumps.pt'
        
        # Use the latest YOLO format for stumps detection
        self.stumps_model = YOLO(stumps_model_path)
        
        # Roboflow API settings for batsman inside/outside detection
        self.API_KEY = "eGBCVKyNP2NYRQpzleUF"
        self.MODEL_ENDPOINT = f"https://detect.roboflow.com/insideoutside/4?api_key={self.API_KEY}"
        
        # Confidence threshold for batsman detection
        self.batsman_detection_threshold = 0.5

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
        """
        Uses Roboflow API to check if batsman is inside or outside the crease
        Returns True if batsman is in crease, False if outside
        """
        batsman_position = self.detect_batsman_position_using_api(frame)
        
        # Debug the response to see all predictions
        print(f"API Response: {batsman_position}")
        
        # If no predictions received, default to in crease (benefit of doubt to batsman)
        if not batsman_position:
            print("No predictions received from API, defaulting to Outside the  crease")
            return False
            
        # Track highest confidence predictions for inside and outside
        highest_inside_conf = 0
        highest_outside_conf = 0
        
        # Process API results
        for pred in batsman_position:
            confidence = pred.get("confidence", 0)
            class_name = pred.get("class", "")
            
            print(f"Prediction: Class={class_name}, Confidence={confidence:.2f}")
            
            # Track highest confidence for each class
            if class_name == "Inside" and confidence > highest_inside_conf:
                highest_inside_conf = confidence
            elif class_name == "Outside" and confidence > highest_outside_conf:
                highest_outside_conf = confidence
        
        # Decision logic - compare highest confidences
        print(f"Highest inside confidence: {highest_inside_conf:.2f}")
        print(f"Highest outside confidence: {highest_outside_conf:.2f}")
        
        # If outside confidence is higher than threshold and higher than inside confidence
        if highest_outside_conf >= self.batsman_detection_threshold and highest_outside_conf > highest_inside_conf:
            print(f"DECISION: Batsman is OUTSIDE crease with confidence {highest_outside_conf:.2f}")
            return False
        else:
            # Either inside has higher confidence, or neither prediction meets threshold
            print(f"DECISION: Batsman is INSIDE crease with confidence {highest_inside_conf:.2f}")
            return True
    
    def detect_batsman_position_using_api(self, frame):
        """
        Detect batsman position in the frame using the Roboflow API
        """
        # Convert frame to PIL image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        
        # Convert image to bytes for API
        buffered = io.BytesIO()
        pil_image.save(buffered, quality=100, format="JPEG")
        
        # Prepare multipart encoder for the API request
        m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})
        
        # Send request to Roboflow API
        try:
            response = requests.post(
                self.MODEL_ENDPOINT,
                data=m,
                headers={'Content-Type': m.content_type}
            )
            
            if response.status_code == 200:
                predictions = response.json().get("predictions", [])
                return predictions
            else:
                print(f"Error: API request failed with status code {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error making API request: {e}")
            return []
