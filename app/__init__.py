# app/__init__.py

from flask import Flask
from .routes import main  # Blueprint for routing

def create_app():
    app = Flask(__name__)
    app.secret_key = 'sankalp_super_secret_🔥'
    
    # Register blueprint
    app.register_blueprint(main)

    return app
