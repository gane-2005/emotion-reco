import numpy as np
import librosa
import soundfile as sf


def extract_features(audio_path, duration=3):
    """
    Extract audio features for emotion classification.
    
    Args:
        audio_path: Path to the audio file
        duration: Duration to load (seconds)
        
    Returns:
        numpy array of extracted features
    """
    try:
        # Load audio file
        audio, sample_rate = librosa.load(audio_path, duration=duration, sr=22050)
        
        # Extract MFCC features (40 coefficients)
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        
        # Extract Chroma features
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        
        # Extract Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        
        # Combine all features
        features = np.concatenate([mfcc, chroma, mel])
        
        return features
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_features_with_augmentation(audio_path, duration=3):
    """
    Extract features with data augmentation for training.
    
    Args:
        audio_path: Path to the audio file
        duration: Duration to load (seconds)
        
    Returns:
        list of feature arrays (original + augmented)
    """
    features_list = []
    
    try:
        # Load audio
        audio, sample_rate = librosa.load(audio_path, duration=duration, sr=22050)
        
        # Original features
        features_list.append(extract_features_from_audio(audio, sample_rate))
        
        # Add noise
        noise = np.random.randn(len(audio))
        audio_with_noise = audio + 0.005 * noise
        features_list.append(extract_features_from_audio(audio_with_noise, sample_rate))
        
        # Time stretch
        audio_stretched = librosa.effects.time_stretch(audio, rate=0.9)
        features_list.append(extract_features_from_audio(audio_stretched, sample_rate))
        
        # Pitch shift
        audio_pitch_shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=2)
        features_list.append(extract_features_from_audio(audio_pitch_shifted, sample_rate))
        
        return features_list
    
    except Exception as e:
        print(f"Error in augmentation: {e}")
        return [extract_features(audio_path, duration)]


def extract_features_from_audio(audio, sample_rate):
    """Extract features from audio array."""
    try:
        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        
        # Chroma
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        
        # Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        
        # Combine
        features = np.concatenate([mfcc, chroma, mel])
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from audio: {e}")
        return None


def analyze_audio_properties(audio_path):
    """
    Analyze audio file properties.
    
    Returns:
        dict with audio properties
    """
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        
        properties = {
            'duration': librosa.get_duration(y=audio, sr=sr),
            'sample_rate': sr,
            'samples': len(audio),
            'rms_energy': float(np.sqrt(np.mean(audio**2))),
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        }
        
        return properties
    
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return None


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
