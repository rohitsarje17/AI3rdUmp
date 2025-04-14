import time
import psutil
import threading
import json
import os
from flask import current_app
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'total_processing_time': 0,
            'model_loading_time': 0,
            'inference_time': 0,
            'api_response_time': 0,
            'frame_processing_times': [],
            'cpu_usage': [],
            'memory_usage': [],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.start_time = None
        self.monitoring = False
        self.monitor_thread = None
        
    def start(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._resource_monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        return self
        
    def stop(self):
        """Stop performance monitoring and calculate total time"""
        if self.start_time:
            self.metrics['total_processing_time'] = time.time() - self.start_time
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)
            return self.metrics
        return None
        
    def _resource_monitor(self):
        """Monitor CPU and memory usage in a separate thread"""
        while self.monitoring:
            self.metrics['cpu_usage'].append(psutil.cpu_percent())
            self.metrics['memory_usage'].append(psutil.virtual_memory().percent)
            time.sleep(0.5)
    
    def log_model_loading_time(self, duration):
        """Log time taken to load models"""
        self.metrics['model_loading_time'] = duration
        
    def log_api_response_time(self, duration):
        """Log time taken for API responses"""
        self.metrics['api_response_time'] += duration
        
    def log_inference_time(self, duration):
        """Log time taken for model inference"""
        self.metrics['inference_time'] += duration
        
    def log_frame_processing_time(self, frame_num, duration):
        """Log time taken to process a specific frame"""
        self.metrics['frame_processing_times'].append({
            'frame': frame_num,
            'time': duration
        })
    
    def save_metrics(self, feature_type):
        """Save metrics to a JSON file"""
        metrics_dir = os.path.join(current_app.config['STATIC_FOLDER'], 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{feature_type}_{timestamp}.json"
        filepath = os.path.join(metrics_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        self._generate_performance_charts(feature_type, timestamp)
        return filename
    
    def _generate_performance_charts(self, feature_type, timestamp):
        """Generate performance visualization charts"""
        metrics_dir = os.path.join(current_app.config['STATIC_FOLDER'], 'metrics')
        charts_dir = os.path.join(metrics_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        # Processing time breakdown chart
        plt.figure(figsize=(10, 6))
        labels = ['Model Loading', 'Inference', 'API Response', 'Other']
        other_time = self.metrics['total_processing_time'] - (
            self.metrics['model_loading_time'] + 
            self.metrics['inference_time'] + 
            self.metrics['api_response_time']
        )
        sizes = [
            self.metrics['model_loading_time'],
            self.metrics['inference_time'],
            self.metrics['api_response_time'],
            other_time
        ]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title(f'Processing Time Breakdown - {feature_type}')
        plt.savefig(os.path.join(charts_dir, f"{feature_type}_{timestamp}_time_breakdown.png"))
        plt.close()
        
        # Resource usage over time
        if len(self.metrics['cpu_usage']) > 0:
            plt.figure(figsize=(12, 6))
            time_points = np.arange(len(self.metrics['cpu_usage'])) * 0.5  # 0.5s intervals
            plt.plot(time_points, self.metrics['cpu_usage'], label='CPU Usage (%)')
            plt.plot(time_points, self.metrics['memory_usage'], label='Memory Usage (%)')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Usage (%)')
            plt.title(f'Resource Usage - {feature_type}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(charts_dir, f"{feature_type}_{timestamp}_resource_usage.png"))
            plt.close()
        
        # Frame processing times
        if len(self.metrics['frame_processing_times']) > 0:
            frames = [item['frame'] for item in self.metrics['frame_processing_times']]
            times = [item['time'] for item in self.metrics['frame_processing_times']]
            
            plt.figure(figsize=(12, 6))
            plt.bar(frames, times)
            plt.xlabel('Frame Number')
            plt.ylabel('Processing Time (seconds)')
            plt.title(f'Frame Processing Times - {feature_type}')
            plt.grid(True, axis='y')
            plt.savefig(os.path.join(charts_dir, f"{feature_type}_{timestamp}_frame_times.png"))
            plt.close()
