from flask import current_app
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
from PIL import Image
import io
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

class NoBallModel:
    def __init__(self):
        # Roboflow API settings for bowler detection
        self.API_KEY = "LIru2IdGcU2dXNmOKBp2"
        self.MODEL_ENDPOINT = f"https://detect.roboflow.com/noball-3/2?api_key={self.API_KEY}"
        
        # Load the no-ball classification model
        no_ball_classification_path = 'E:\\BTech\\Project\\third-umpire-decision\\trainedModels\\noBall.pt'
        self.classification_model = YOLO(no_ball_classification_path)
        
        # Confidence thresholds
        self.bowler_detection_threshold = 0.4
        self.no_ball_threshold = 0.65
        self.target_class = "bowler"  # Class name for bowler detection
        
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        is_no_ball = False
        frame_count = 0
        no_ball_frame = None
        frames = []
        
        # Create temporary folder for cropped images
        temp_folder = current_app.config['UPLOAD_FOLDER'] + '/temp'
        os.makedirs(temp_folder, exist_ok=True)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Step 1: Detect bowler in the frame using Roboflow API
            bowler_detections = self.detect_bowler_using_api(frame)
            
            for pred in bowler_detections:
                if pred["class"] == self.target_class and pred["confidence"] >= self.bowler_detection_threshold:
                    # Get bounding box coordinates from the API response
                    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                    
                    # Calculate image dimensions
                    img_height, img_width = frame.shape[:2]
                    
                    # Calculate bounding box coordinates (API returns center x,y and width/height)
                    x1 = int(x - w/2)
                    y1 = int(y - h/2)
                    x2 = int(x + w/2)
                    y2 = int(y + h/2)
                    
                    # Add padding to include the bowling action
                    padding_w = int(w * 0.55)
                    padding_h = int(h * 0.25)
                    
                    # Ensure coordinates stay within frame boundaries
                    x1_pad = max(0, x1 - padding_w)
                    y1_pad = max(0, y1 - padding_h)
                    x2_pad = min(img_width, x2 + padding_w)
                    y2_pad = min(img_height, y2 + padding_h)
                    
                    # Crop and resize the bowler area
                    cropped_bowler = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                    if cropped_bowler.size == 0:
                        continue
                        
                    cropped_resized = cv2.resize(cropped_bowler, (224, 224))
                    
                    # Save the cropped image temporarily
                    cropped_image_path = os.path.join(temp_folder, f"frame_{frame_count}.jpg")
                    cv2.imwrite(cropped_image_path, cropped_resized)
                    
                    # Step 2: Classify cropped image to detect no ball
                    classification_results = self.classification_model(cropped_image_path)
                    
                    # Process classification results (assuming class 0 = normal, class 1 = no-ball)
                    if hasattr(classification_results[0], "probs") and classification_results[0].probs is not None:
                        no_ball_confidence = classification_results[0].probs.data[1].item()  # Index 1 for no-ball
                        print(f"No-Ball Confidence: {no_ball_confidence:.2f}")
                        
                        if no_ball_confidence >= self.no_ball_threshold:
                            print(f"No-ball detected at frame {frame_count} with confidence {no_ball_confidence:.2f}")
                            no_ball_frame = frame.copy()
                            
                            # Draw bounding box and label
                            cv2.rectangle(no_ball_frame, (x1_pad, y1_pad), (x2_pad, y2_pad), (0, 0, 255), 2)
                            cv2.putText(no_ball_frame, f"No-Ball: {no_ball_confidence:.2f}", 
                                        (x1_pad, y1_pad-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.9, (0, 0, 255), 2)
                            
                            is_no_ball = True
                            break
            
            if is_no_ball:
                break
                
            # Save the current frame
            frame_filename = f"frame_{frame_count}.jpg"
            frame_path = current_app.config['STATIC_FOLDER'] + '/' + frame_filename
            cv2.imwrite(frame_path, frame)
            frames.append({'filename': frame_filename, 'label': f"Frame {frame_count}"})
        
        cap.release()
        
        # Clean up temporary folder
        for file in os.listdir(temp_folder):
            os.remove(os.path.join(temp_folder, file))
        os.rmdir(temp_folder)
        
        # Save and add the no ball frame if found
        if no_ball_frame is not None:
            no_ball_frame_rgb = cv2.cvtColor(no_ball_frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(current_app.config['UPLOAD_FOLDER'] + '/no_ball_frame.jpg', no_ball_frame)
            cv2.imwrite(current_app.config['STATIC_FOLDER'] + '/no_ball_frame.jpg', no_ball_frame)
            
            # Create a plot with the detection
            plt.figure(figsize=(10, 8))
            plt.imshow(no_ball_frame_rgb)
            plt.title('No Ball Detected')
            plt.axis('off')
            plt.savefig(current_app.config['STATIC_FOLDER'] + '/no_ball_frame_plot.jpg')
            plt.close()
            
            frames.append({'filename': 'no_ball_frame.jpg', 'label': 'No Ball Frame'})
            
        return is_no_ball, frames
        
    def detect_bowler_using_api(self, frame):
        """
        Detect bowler in the frame using the Roboflow API
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