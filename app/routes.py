from flask import Blueprint, render_template, request, redirect, url_for, current_app, jsonify
import os
import time
from werkzeug.utils import secure_filename
from .models.no_ball_model import NoBallModel
from .models.runout_stumping_model import RunoutStumpingModel
from .models.caught_behind_model import CaughtBehindModel
from .utils.performance import PerformanceMonitor

main = Blueprint('main', __name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return redirect(request.url)
        
    video = request.files['video']
    feature = request.form.get('feature')
    
    if video.filename == '':
        return redirect(request.url)
        
    if video and allowed_file(video.filename):
        # Start performance monitoring
        monitor = PerformanceMonitor().start()
        
        filename = secure_filename(video.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        video.save(filepath)
        
        model_load_start = time.time()
        if feature == 'no_ball':
            model = NoBallModel()
            model_load_end = time.time()
            monitor.log_model_loading_time(model_load_end - model_load_start)
            
            inference_start = time.time()
            is_no_ball, frames = model.process_video(filepath)
            inference_end = time.time()
            monitor.log_inference_time(inference_end - inference_start)
            
            # Change the result display for no ball detection
            result = "No Ball" if is_no_ball else "Legal Ball"
        elif feature in ['runout', 'stumping']:
            model = RunoutStumpingModel()
            model_load_end = time.time()
            monitor.log_model_loading_time(model_load_end - model_load_start)
            
            inference_start = time.time()
            is_out, frames = model.process_video(filepath)
            inference_end = time.time()
            monitor.log_inference_time(inference_end - inference_start)
            
            result = "OUT" if is_out else "NOT OUT"
        elif feature == 'caught_behind':
            model = CaughtBehindModel()
            model_load_end = time.time()
            monitor.log_model_loading_time(model_load_end - model_load_start)
            
            inference_start = time.time()
            is_out, frames = model.process_video(filepath)
            inference_end = time.time()
            monitor.log_inference_time(inference_end - inference_start)
            
            result = "OUT" if is_out else "NOT OUT"
        else:
            return redirect(request.url)
        
        # Stop performance monitoring and save metrics
        metrics = monitor.stop()
        metrics_file = monitor.save_metrics(feature)
        
        return render_template('index.html', result=result, frames=frames, feature=feature, 
                              performance_metrics=metrics, metrics_file=metrics_file)
        
    return redirect(request.url)

@main.route('/performance')
def performance():
    # Get all performance metric files
    metrics_dir = os.path.join(current_app.config['STATIC_FOLDER'], 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_files = []
    for filename in os.listdir(metrics_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(metrics_dir, filename)
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                    feature_type = filename.split('_')[0]
                    metrics_files.append({
                        'filename': filename,
                        'feature': feature_type,
                        'timestamp': data.get('timestamp', 'Unknown'),
                        'total_time': data.get('total_processing_time', 0)
                    })
                except:
                    continue
    
    return render_template('performance.html', metrics_files=metrics_files)

@main.route('/performance/<filename>')
def performance_detail(filename):
    metrics_dir = os.path.join(current_app.config['STATIC_FOLDER'], 'metrics')
    filepath = os.path.join(metrics_dir, filename)
    
    if not os.path.exists(filepath):
        return redirect(url_for('main.performance'))
    
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    
    feature_type = filename.split('_')[0]
    timestamp = '_'.join(filename.split('_')[1:]).replace('.json', '')
    
    # Check for chart files
    charts = {
        'time_breakdown': f"{feature_type}_{timestamp}_time_breakdown.png",
        'resource_usage': f"{feature_type}_{timestamp}_resource_usage.png",
        'frame_times': f"{feature_type}_{timestamp}_frame_times.png"
    }
    
    for chart_key, chart_file in charts.items():
        if not os.path.exists(os.path.join(metrics_dir, 'charts', chart_file)):
            charts[chart_key] = None
    
    return render_template('performance_detail.html', 
                          metrics=metrics, 
                          filename=filename,
                          feature=feature_type,
                          charts=charts)

@main.route('/run_benchmark', methods=['POST'])
def run_benchmark():
    feature = request.form.get('feature')
    test_video = request.form.get('test_video')
    iterations = int(request.form.get('iterations', 1))
    
    if not feature or not test_video:
        return jsonify({'error': 'Missing required parameters'})
    
    video_path = os.path.join(current_app.config['UPLOAD_FOLDER'], test_video)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Test video not found'})
    
    results = []
    
    for i in range(iterations):
        monitor = PerformanceMonitor().start()
        
        model_load_start = time.time()
        if feature == 'no_ball':
            model = NoBallModel()
            model_load_end = time.time()
            monitor.log_model_loading_time(model_load_end - model_load_start)
            
            inference_start = time.time()
            is_no_ball, frames = model.process_video(video_path)
            inference_end = time.time()
            monitor.log_inference_time(inference_end - inference_start)
            
        elif feature in ['runout', 'stumping']:
            model = RunoutStumpingModel()
            model_load_end = time.time()
            monitor.log_model_loading_time(model_load_end - model_load_start)
            
            inference_start = time.time()
            is_out, frames = model.process_video(video_path)
            inference_end = time.time()
            monitor.log_inference_time(inference_end - inference_start)
            
        elif feature == 'caught_behind':
            model = CaughtBehindModel()
            model_load_end = time.time()
            monitor.log_model_loading_time(model_load_end - model_load_start)
            
            inference_start = time.time()
            is_out, frames = model.process_video(video_path)
            inference_end = time.time()
            monitor.log_inference_time(inference_end - inference_start)
        
        metrics = monitor.stop()
        metrics_file = monitor.save_metrics(f"{feature}_benchmark_{i+1}")
        results.append({
            'iteration': i+1,
            'metrics': metrics,
            'metrics_file': metrics_file
        })
    
    # Calculate average metrics
    avg_metrics = {
        'total_processing_time': sum(r['metrics']['total_processing_time'] for r in results) / iterations,
        'model_loading_time': sum(r['metrics']['model_loading_time'] for r in results) / iterations,
        'inference_time': sum(r['metrics']['inference_time'] for r in results) / iterations,
        'api_response_time': sum(r['metrics']['api_response_time'] for r in results) / iterations,
    }
    
    return jsonify({
        'success': True,
        'iterations': iterations,
        'feature': feature,
        'results': results,
        'average_metrics': avg_metrics
    })