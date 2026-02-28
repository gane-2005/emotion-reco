import os
import sys

# Append backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

import torch
import numpy as np

from backend.utils.feature_extraction import extract_features
from backend.model.model_pytorch import get_model
import backend.config as config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=len(config.EMOTIONS), input_size=config.COMBINED_FEATURE_DIM)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
model.to(device)
model.eval()

output_dir = r"d:\learnthonproject\speech-emotion-analysis\REAL_HUMAN_SAMPLES"
for f in os.listdir(output_dir):
    if f.endswith(".wav"):
        p = os.path.join(output_dir, f)
        features = extract_features(p, fixed_length=config.FIXED_LENGTH)
        if features is not None:
            tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                idx = int(np.argmax(probs))
                print(f"{f}: Predicted -> {config.EMOTIONS[idx]} ({probs[idx]*100:.1f}%)")
