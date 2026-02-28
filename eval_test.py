import os
import sys
import torch
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from backend.utils.feature_extraction import extract_features
from backend.model.model_pytorch import get_model
import backend.config as config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=len(config.EMOTIONS), input_size=config.COMBINED_FEATURE_DIM)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
model.to(device)

def predict(audio_path, eval_mode=True):
    features = extract_features(audio_path, fixed_length=config.FIXED_LENGTH)
    if features is None: return None
    
    input_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    if eval_mode:
        model.eval()
    else:
        model.train()
        
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
    return probabilities

# Test with an angry file
angry_path = r"d:\learnthonproject\speech-emotion-analysis\dataset\RAVDESS\Actor_01\03-01-05-01-01-01-01.wav"

probs_eval = predict(angry_path, eval_mode=True)
print("Eval mode probabilities:", [f"{config.EMOTIONS[i]}: {probs_eval[i]:.4f}" for i in range(4)])

probs_train = predict(angry_path, eval_mode=False)
print("Train mode probabilities:", [f"{config.EMOTIONS[i]}: {probs_train[i]:.4f}" for i in range(4)])
