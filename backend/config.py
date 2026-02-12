import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Model configuration
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'emotion_model.pth')
# We transitioned to log-mel spectrograms; scikit-learn scaler is no longer used.

# Database configuration
DATABASE_PATH = os.path.join(BASE_DIR, 'emotions.db')

# API configuration
API_HOST = '0.0.0.0'
API_PORT = 5000
DEBUG = True

# Emotion labels - Unified order for model and UI
EMOTIONS = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust', 'Neutral']

# CORS configuration
CORS_ORIGINS = ['http://localhost:5173', 'http://localhost:3000', 'http://127.0.0.1:5173']

# Preprocessing Constants
TARGET_SAMPLE_RATE = 16000
FIXED_LENGTH = 200

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
