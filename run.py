from flask import Flask
from app import create_app

app = create_app()

if __name__ == '__main__':
    # Host on all available interfaces (0.0.0.0)
    # This allows access from other devices on the network
    app.run(host='0.0.0.0', debug=True)