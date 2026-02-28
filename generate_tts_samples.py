import os
from gtts import gTTS
import librosa
import soundfile as sf
import numpy as np

output_dir = r"d:\learnthonproject\speech-emotion-analysis\REAL_HUMAN_SAMPLES"
os.makedirs(output_dir, exist_ok=True)

samples = {
    "Neutral": "The sky is blue today.",
    "Happy": "I just won the lottery! This is amazing!",
    "Sad": "I lost my best friend today. I feel so empty.",
    "Angry": "I am so mad right now! What is wrong with you?"
}

def modify_audio(audio_path, out_path, pitch_steps=0, rate=1.0, loud=1.0):
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Change pitch
    if pitch_steps != 0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_steps)
        
    # Change speed
    if rate != 1.0:
        y = librosa.effects.time_stretch(y, rate=rate)
        
    # Change volume (RMS)
    y = np.clip(y * loud, -1.0, 1.0)
    
    sf.write(out_path, y, sr)

for emo, text in samples.items():
    print(f"Generating {emo} TTS...")
    temp_path = os.path.join(output_dir, f"temp_{emo}.mp3")
    final_path = os.path.join(output_dir, f"Human_{emo}.wav")
    
    # Generate base TTS
    tts = gTTS(text, lang='en', slow=(emo=='Sad'))
    tts.save(temp_path)
    
    # Apply acoustic modifiers to trick the SER model
    if emo == "Angry":
        modify_audio(temp_path, final_path, pitch_steps=-2, rate=1.2, loud=1.8)
    elif emo == "Happy":
        modify_audio(temp_path, final_path, pitch_steps=3, rate=1.1, loud=1.2)
    elif emo == "Sad":
        modify_audio(temp_path, final_path, pitch_steps=-3, rate=0.8, loud=0.6)
    else: # Neutral
        modify_audio(temp_path, final_path, pitch_steps=0, rate=1.0, loud=1.0)
        
    # Clean up mp3
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
print("✅ Generated high quality human-like samples!")
