from flask import Flask
import os

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'E:\\BTech\\Project\\third-umpire-decision\\uploads'
    app.config['STATIC_FOLDER'] = 'E:\\BTech\\Project\\third-umpire-decision\\app\\static'

    # Ensure the upload directory exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['STATIC_FOLDER']):
        os.makedirs(app.config['STATIC_FOLDER'])

    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app