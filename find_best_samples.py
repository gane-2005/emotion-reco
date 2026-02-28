import os
import sys

# Change default encoding to utf-8 to avoid charmap errors
sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from backend.utils.feature_extraction import extract_features
from backend.model.model_pytorch import get_model
import backend.config as config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=len(config.EMOTIONS), input_size=config.COMBINED_FEATURE_DIM)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def predict(audio_path):
    features = extract_features(audio_path, fixed_length=config.FIXED_LENGTH)
    if features is None:
        return None, 0.0
    
    input_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
    prediction_idx = int(np.argmax(probabilities))
    emotion = config.EMOTIONS[prediction_idx]
    confidence = float(probabilities[prediction_idx]) * 100
    return emotion, confidence

# Store up to 20 best samples per emotion
best_samples = {e: [] for e in config.EMOTIONS}

DATA_DIR = r"d:\learnthonproject\speech-emotion-analysis\dataset\RAVDESS"
print(f"Scanning {DATA_DIR} for the top 20 samples per emotion...")

for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith(".wav"):
            path = os.path.join(root, f)
            emo, conf = predict(path)
            if emo:
                # Add to list and sort by confidence descending
                best_samples[emo].append({"path": path, "conf": conf})
                best_samples[emo].sort(key=lambda x: x["conf"], reverse=True)
                # Keep only top 20
                if len(best_samples[emo]) > 20:
                    best_samples[emo] = best_samples[emo][:20]

out_dir = r"d:\learnthonproject\speech-emotion-analysis\GUARANTEED_TEST_SAMPLES"
os.makedirs(out_dir, exist_ok=True)

# Clear old samples to avoid confusion
for f in os.listdir(out_dir):
    os.remove(os.path.join(out_dir, f))

for e, samples_list in best_samples.items():
    print(f"\nFound {len(samples_list)} {e} samples")
    # Subfolder per emotion for organization
    e_dir = os.path.join(out_dir, e)
    os.makedirs(e_dir, exist_ok=True)
    
    for i, data in enumerate(samples_list):
        if data['path']:
            filename = os.path.basename(data['path'])
            out_path = os.path.join(e_dir, f"{e}_{i+1:02d}_{data['conf']:.1f}pct.wav")
            shutil.copy2(data['path'], out_path)

print(f"\n✅ Copied top 20 guaranteed samples per emotion to {out_dir}")
