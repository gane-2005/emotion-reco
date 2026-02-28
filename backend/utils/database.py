import sqlite3
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE_PATH


def init_database():
    """Initialize the database and create tables if they don't exist."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            filename TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DATABASE_PATH}")


def save_prediction(emotion, confidence, filename=None):
    """Save a prediction to the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions (emotion, confidence, filename, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (emotion, confidence, filename, datetime.now()))
    
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return prediction_id


def get_all_predictions():
    """Retrieve all predictions from the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, emotion, confidence, filename, timestamp
        FROM predictions
        ORDER BY timestamp DESC
    ''')
    
    predictions = []
    for row in cursor.fetchall():
        predictions.append({
            'id': row[0],
            'emotion': row[1],
            'confidence': row[2],
            'filename': row[3],
            'timestamp': row[4]
        })
    
    conn.close()
    return predictions


def delete_prediction(prediction_id):
    """Delete a prediction from the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM predictions WHERE id = ?', (prediction_id,))
    
    conn.commit()
    conn.close()


def get_stats():
    """Get statistics for the mood dashboard."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Total predictions
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total = cursor.fetchone()[0]
    
    # Predictions today
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE date(timestamp) = date('now', 'localtime')")
    today = cursor.fetchone()[0]
    
    # Predictions this week
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE date(timestamp) >= date('now', 'localtime', '-7 days')")
    this_week = cursor.fetchone()[0]
    
    # Most common emotion
    cursor.execute('''
        SELECT emotion, COUNT(*) as count 
        FROM predictions 
        GROUP BY emotion 
        ORDER BY count DESC 
        LIMIT 1
    ''')
    row = cursor.fetchone()
    most_common = row[0] if row else None
    
    # Emotion distribution
    cursor.execute('''
        SELECT emotion, COUNT(*) as count, AVG(confidence) as avg_conf 
        FROM predictions 
        GROUP BY emotion 
        ORDER BY count DESC
    ''')
    distribution = []
    for row in cursor.fetchall():
        distribution.append({
            'emotion': row[0],
            'count': row[1],
            'avg_confidence': round(row[2], 2)
        })
    
    conn.close()
    
    return {
        'success': True,
        'total': total,
        'today': today,
        'this_week': this_week,
        'most_common_emotion': most_common,
        'distribution': distribution
    }


if __name__ == '__main__':
    init_database()
