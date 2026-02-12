import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Model configuration
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'emotion_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')

# Database configuration
DATABASE_PATH = os.path.join(os.path.dirname(BASE_DIR), 'emotions.db')

# API configuration
API_HOST = '0.0.0.0'
API_PORT = 5000
DEBUG = True

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# CORS configuration
CORS_ORIGINS = ['http://localhost:5173', 'http://localhost:3000', 'http://127.0.0.1:5173']

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
