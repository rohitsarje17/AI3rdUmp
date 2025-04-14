from flask import Blueprint, render_template, request, redirect, url_for, current_app
import os
from werkzeug.utils import secure_filename
from .models.no_ball_model import NoBallModel
from .models.runout_stumping_model import RunoutStumpingModel
from .models.caught_behind_model import CaughtBehindModel

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
        filename = secure_filename(video.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        video.save(filepath)
        
        if feature == 'no_ball':
            model = NoBallModel()
            is_no_ball, frames = model.process_video(filepath)
            # Change the result display for no ball detection
            result = "No Ball" if is_no_ball else "Legal Ball"
        elif feature in ['runout', 'stumping']:
            model = RunoutStumpingModel()
            is_out, frames = model.process_video(filepath)
            result = "OUT" if is_out else "NOT OUT"
        elif feature == 'caught_behind':
            model = CaughtBehindModel()
            is_out, frames = model.process_video(filepath)
            result = "OUT" if is_out else "NOT OUT"
        else:
            return redirect(request.url)
            
        return render_template('index.html', result=result, frames=frames, feature=feature)
        
    return redirect(request.url)