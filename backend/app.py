from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

import config
from utils.database import init_database, save_prediction, get_all_predictions, delete_prediction
from utils.feature_extraction import extract_features
import joblib
import traceback

app = Flask(__name__)
CORS(app, origins=config.CORS_ORIGINS)

# Initialize database
init_database()

# Load model and scaler (will be created after training)
model = None
scaler = None

try:
    if os.path.exists(config.MODEL_PATH) and os.path.exists(config.SCALER_PATH):
        model = joblib.load(config.MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
        print("✅ Model and scaler loaded successfully")
    else:
        print("⚠️  Model not found. Please train the model first using train_model.py")
except Exception as e:
    print(f"⚠️  Error loading model: {e}")


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model_loaded = model is not None and scaler is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'message': 'Speech Emotion Analysis API is running' if model_loaded else 'Model not loaded yet'
    })


@app.route('/api/predict', methods=['POST'])
def predict_emotion():
    """Predict emotion from uploaded audio file."""
    try:
        # Check if model is loaded
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.',
                'success': False
            }), 500
        
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({
                'error': 'No audio file provided',
                'success': False
            }), 400
        
        file = request.files['audio']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {", ".join(config.ALLOWED_EXTENSIONS)}',
                'success': False
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"DEBUG: Saved file to {filepath}, Size: {os.path.getsize(filepath)} bytes")
        
        
        try:
            # Extract features
            features = extract_features(filepath)
            
            if features is None:
                return jsonify({
                    'error': 'Failed to extract features from audio',
                    'success': False
                }), 400
            
            # Scale features
            features_scaled = scaler.transform([features])
            
            # Predict emotion
            prediction = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled)
            
            emotion = prediction[0]
            confidence = float(max(probabilities[0])) * 100
            
            # Save to database
            prediction_id = save_prediction(emotion, confidence, filename)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'emotion': emotion,
                'confidence': round(confidence, 2),
                'prediction_id': prediction_id,
                'all_probabilities': {
                    config.EMOTIONS[i]: round(float(probabilities[0][i]) * 100, 2)
                    for i in range(len(config.EMOTIONS))
                }
            })
        
        except Exception as e:
            # Clean up file if error occurs
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get all prediction history."""
    try:
        predictions = get_all_predictions()
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions)
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/history/<int:prediction_id>', methods=['DELETE'])
def delete_history_item(prediction_id):
    """Delete a specific prediction from history."""
    try:
        delete_prediction(prediction_id)
        return jsonify({
            'success': True,
            'message': f'Prediction {prediction_id} deleted'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/emotions', methods=['GET'])
def get_emotions():
    """Get list of supported emotions."""
    return jsonify({
        'success': True,
        'emotions': config.EMOTIONS
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎤 Speech Emotion Analysis Using Voice - Backend API")
    print("="*60)
    print(f"Server running at: http://{config.API_HOST}:{config.API_PORT}")
    print(f"Upload folder: {config.UPLOAD_FOLDER}")
    print(f"Database: {config.DATABASE_PATH}")
    print("="*60 + "\n")
    
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.DEBUG
    )
