from flask import current_app
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

class CaughtBehindModel:
    def __init__(self):
        spike_model_path = 'E:\\BTech\\Project\\third-umpire-decision\\trainedModels\\spike.pt'
        edge_model_path = 'E:\\BTech\\Project\\third-umpire-decision\\trainedModels\\edge.pt'

        self.spike_model = YOLO(spike_model_path)
        self.edge_model = YOLO(edge_model_path)
        self.spike_label = "spike"

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        spike_detected = False
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            results = self.spike_model(frame)
            detected_spike = False

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    label = self.spike_model.names[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if label == self.spike_label:
                        print(f"Spike detected at frame {frame_count}")

                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_rgb, "Spike", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        edge_results = self.edge_model(frame)
                        decision_array = [self.edge_model.names[int(box.cls[0])] for result in edge_results for box in result.boxes]

                        for edge_result in edge_results:
                            for edge_box in edge_result.boxes:
                                ex1, ey1, ex2, ey2 = map(int, edge_box.xyxy[0])
                                edge_label = self.edge_model.names[int(edge_box.cls[0])]

                                cv2.rectangle(frame_rgb, (ex1, ey1), (ex2, ey2), (255, 0, 0), 2)
                                cv2.putText(frame_rgb, edge_label, (ex1, ey1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        if "noedge" in decision_array:
                            decision_text = "Clear Gap between bat and ball, NOT OUT."
                            color = (255, 0, 0)
                            spike_detected = False
                        else:
                            decision_text = "Edge detected, OUT."
                            color = (255, 255, 0)
                            spike_detected = True

                        print(decision_text)
                        cv2.putText(frame_rgb, decision_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                        # Save the frame to both upload and static folders
                        cv2.imwrite(current_app.config['UPLOAD_FOLDER'] + '/spike_detected_frame.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(current_app.config['STATIC_FOLDER'] + '/spike_detected_frame.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                        
                        # Create and save plot
                        plt.figure(figsize=(10, 8))
                        plt.imshow(frame_rgb)
                        plt.title(decision_text)
                        plt.axis('off')
                        plt.savefig(current_app.config['STATIC_FOLDER'] + '/spike_detected_frame_plot.jpg')
                        plt.close()  # Close the figure to avoid memory leaks

                        frames.append({'filename': 'spike_detected_frame.jpg', 'label': 'Spike Detected Frame'})
                        cap.release()
                        break

                if spike_detected:
                    break

            frame_filename = f"frame_{frame_count}.jpg"
            frame_path = current_app.config['STATIC_FOLDER'] + '/' + frame_filename
            cv2.imwrite(frame_path, frame)
            frames.append({'filename': frame_filename, 'label': f"Frame {frame_count}"})

            if spike_detected:
                break

        if not spike_detected:
            print("No spike detected in entire video, NOT OUT")

        cap.release()
        return spike_detected, frames
