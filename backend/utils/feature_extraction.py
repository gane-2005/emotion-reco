import numpy as np
import librosa
import soundfile as sf
import os
from pydub import AudioSegment
import io

def preprocess_audio(audio_path, target_sr=16000):
    """
    Load and preprocess audio: resample, convert to mono, trim silence, and normalize.
    """
    try:
        # Load audio with librosa
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio)
        
        # Normalize
        if len(audio) > 0:
            audio = librosa.util.normalize(audio)
            
        return audio, sr
    except Exception as e:
        print(f"Error preprocessing audio {audio_path}: {e}")
        return None, None

def extract_log_mel_spectrogram(audio, sr=16000, n_mels=128, n_fft=2048, hop_length=512):
    """
    Extract Log-Mel Spectrogram features.
    """
    try:
        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, 
                                                 n_fft=n_fft, hop_length=hop_length)
        
        # Convert to log scale (power to db)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    except Exception as e:
        print(f"Error extracting Mel Spectrogram: {e}")
        return None

def extract_features(audio_path, target_sr=16000, fixed_length=None):
    """
    Unified feature extraction for both inference and training.
    """
    audio, sr = preprocess_audio(audio_path, target_sr=target_sr)
    if audio is None:
        return None
    
    # Extract Log-Mel Spectrogram
    mel_spec = extract_log_mel_spectrogram(audio, sr=sr)
    
    if fixed_length and mel_spec is not None:
        # Pad or truncate to a fixed sequence length
        if mel_spec.shape[1] < fixed_length:
            pad_width = fixed_length - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0,0), (0, pad_width)), mode='constant')
        else:
            mel_spec = mel_spec[:, :fixed_length]
            
    return mel_spec

# Data Augmentation Methods
def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def time_stretch(audio, rate=1.0):
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def extract_features_with_augmentation(audio_path, target_sr=16000):
    """
    Returns a list of augmented features for training.
    """
    audio, sr = preprocess_audio(audio_path, target_sr=target_sr)
    if audio is None:
        return []
    
    aug_list = []
    
    # Original
    aug_list.append(extract_log_mel_spectrogram(audio, sr=sr))
    
    # Add Noise
    aug_list.append(extract_log_mel_spectrogram(add_noise(audio), sr=sr))
    
    # Time Stretch
    aug_list.append(extract_log_mel_spectrogram(time_stretch(audio, rate=0.8), sr=sr))
    aug_list.append(extract_log_mel_spectrogram(time_stretch(audio, rate=1.2), sr=sr))
    
    # Pitch Shift
    aug_list.append(extract_log_mel_spectrogram(pitch_shift(audio, sr, n_steps=2), sr=sr))
    aug_list.append(extract_log_mel_spectrogram(pitch_shift(audio, sr, n_steps=-2), sr=sr))
    
    # Filter out None results
    return [aug for aug in aug_list if aug is not None]

def convert_to_wav(file_stream, output_path):
    """
    Convert WebM or other formats to WAV using pydub.
    """
    try:
        audio = AudioSegment.from_file(file_stream)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Error converting audio to WAV: {e}")
        return False


if __name__ == '__main__':
    # Test feature extraction
    import sys
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        features = extract_features(audio_file)
        if features is not None:
            print(f"✅ Extracted {len(features)} features")
            print(f"Feature shape: {features.shape}")
        else:
            print("❌ Feature extraction failed")
