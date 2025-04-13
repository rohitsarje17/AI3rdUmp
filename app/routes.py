from flask import Blueprint, render_template, request, redirect, url_for, current_app
from .models.no_ball_model import NoBallModel
from .models.runout_stumping_model import RunoutStumpingModel
from .models.caught_behind_model import CaughtBehindModel
import os
from werkzeug.utils import secure_filename

main = Blueprint('main', __name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/process', methods=['POST'])
def process():
    if 'video' not in request.files:
        return redirect(request.url)
    
    video = request.files['video']
    if video.filename == '':
        return redirect(request.url)
    
    if video and allowed_file(video.filename):
        filename = secure_filename(video.filename)
        video_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure the upload directory exists
        if not os.path.exists(current_app.config['UPLOAD_FOLDER']):
            os.makedirs(current_app.config['UPLOAD_FOLDER'])
        
        print(f"Saving video to: {video_path}")  # Debug statement
        video.save(video_path)
        print(f"Video saved successfully to: {video_path}")  # Debug statement

        feature = request.form.get('feature')
        
        if feature == 'no_ball':
            model = NoBallModel()
            result, frames = model.process_video(video_path)
        elif feature == 'runout_stumping':
            model = RunoutStumpingModel()
            result, frames = model.process_video(video_path)
        elif feature == 'caught_behind':
            model = CaughtBehindModel()
            result, frames = model.process_video(video_path)
        else:
            result = "Invalid feature selected."
            frames = []

        decision = "Out" if result else "Not Out"
        return render_template('index.html', result=decision, frames=frames)
    
    return redirect(request.url)